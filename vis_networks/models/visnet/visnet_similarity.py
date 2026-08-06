import torch
import torch.nn as nn
import torch.nn.functional as F
from .transform import get_transformer
from .visnet import Model as VisNetModel

CUSTOM_TRANSFORM = get_transformer

FEAT_DIM = 3072   # 1024 (fc_1) + 2048 (fc_2) from the frozen backbone
PROJ_DIM = 128


class Model(nn.Module):
    def __init__(self, n_classes, input_size, mean=None, std=None,
                 dropout_p=0.4, pretrained_visnet_path=None, **kwargs):
        super().__init__()
        # input_size is [2, 3, 3, H, W]; base VisNet expects [3, 3, H, W]
        single_size = list(input_size[1:])
        single_mean = mean[0] if (mean is not None and mean.dim() >= 2 and mean.size(0) == 2) else mean
        single_std  = std[0]  if (std  is not None and std.dim()  >= 2 and std.size(0)  == 2) else std

        base = VisNetModel(n_classes, single_size, mean=single_mean, std=single_std, dropout_p=dropout_p)
        # Trigger LazyLinear initialization
        with torch.no_grad():
            base(torch.zeros(1, *single_size))

        if pretrained_visnet_path:
            sd = torch.load(pretrained_visnet_path, weights_only=True, map_location='cpu')
            base.load_state_dict(sd, strict=False)

        # Copy normalization buffers from base model (already shaped [1, 3, 3, 1, 1])
        self.register_buffer('mean', base.mean.clone())
        self.register_buffer('std',  base.std.clone())

        # Frozen backbone: conv streams + both FC projection layers
        self.stream_1 = base.stream_1
        self.stream_2 = base.stream_2
        self.stream_3 = base.stream_3
        self.fc_1 = base.fc[0]   # → 1024-dim
        self.fc_2 = base.fc[1]   # → 2048-dim
        del base

        for p in self.stream_1.parameters(): p.requires_grad = False
        for p in self.stream_2.parameters(): p.requires_grad = False
        for p in self.stream_3.parameters(): p.requires_grad = False
        for p in self.fc_1.parameters():     p.requires_grad = False
        for p in self.fc_2.parameters():     p.requires_grad = False

        # Shared projector: maps 3072-dim backbone features → 128-dim embedding
        self.projector = nn.Sequential(
            nn.Linear(FEAT_DIM, PROJ_DIM),
            nn.ReLU(True),
        )

        # Classification head: cat([p_curr, p_diff]) → n_classes
        self.cls_head = nn.Sequential(
            nn.Linear(PROJ_DIM * 2, 64),
            nn.ReLU(True),
            nn.Dropout(dropout_p),
            nn.Linear(64, n_classes),
        )

        # Regression head: [cos_sim, ||f_diff||, ||f_curr||] → visibility (1-10 mi)
        self.vis_head = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(True),
            nn.Linear(32, 1),
        )

    def _extract_features(self, x):
        """Run the frozen backbone on a [B, 3, 3, H, W] input; return [B, 3072]."""
        inputs = x.split(1, 1)
        orig    = inputs[0].squeeze(1)
        pc      = inputs[1].squeeze(1)
        fft_in  = inputs[2].squeeze(1)

        s1 = self.stream_1[0](fft_in)
        s2 = self.stream_2[0](pc)
        s3 = self.stream_3[0](orig)
        s1 = torch.add(s1, torch.add(s2, s3))

        s1 = self.stream_1[1](s1)
        s2 = self.stream_2[1](s2)
        s3 = self.stream_3[1](s3)
        s1 = s1.add(s2.add(s3))

        s1 = self.stream_1[2](s1)
        s2 = self.stream_2[2](s2)
        s3 = self.stream_3[2](s3)

        f1  = self.fc_1(s1)           # [B, 1024]
        f23 = self.fc_2(s2.add(s3))   # [B, 2048]
        return torch.cat([f1, f23], dim=1)   # [B, 3072]

    def forward(self, x):
        # x: [B, 2, 3, 3, H, W]  (current @ dim-1 index 0, reference @ index 1)
        current   = x[:, 0]   # [B, 3, 3, H, W]
        reference = x[:, 1]   # [B, 3, 3, H, W]

        current   = current.subtract(self.mean).divide(self.std)
        reference = reference.subtract(self.mean).divide(self.std)

        # Frozen backbone: no gradient storage needed
        with torch.no_grad():
            f_curr = self._extract_features(current)    # [B, 3072]
            f_ref  = self._extract_features(reference)  # [B, 3072]

        f_diff = torch.abs(f_curr - f_ref)   # [B, 3072]

        # Project into shared embedding space
        p_curr = self.projector(f_curr)    # [B, 128]
        p_ref  = self.projector(f_ref)     # [B, 128]
        p_diff = self.projector(f_diff)    # [B, 128]

        # Cosine similarity between projected current and reference embeddings
        cos_sim = F.cosine_similarity(p_curr, p_ref, dim=1)   # [B]

        # Classification: concat current + diff projections
        cls_logits = self.cls_head(torch.cat([p_curr, p_diff], dim=1))   # [B, n_classes]

        # Regression: physical signal — similarity, absolute diff magnitude, current magnitude
        diff_norm = torch.norm(f_diff, dim=1, keepdim=True)   # [B, 1]
        curr_norm = torch.norm(f_curr, dim=1, keepdim=True)   # [B, 1]
        # Normalize norms to O(1) so vis_head input doesn't saturate sigmoid
        diff_norm_n = diff_norm / (diff_norm.detach().mean().clamp(min=1.0))
        curr_norm_n = curr_norm / (curr_norm.detach().mean().clamp(min=1.0))
        reg_in    = torch.cat([cos_sim.unsqueeze(1), diff_norm_n, curr_norm_n], dim=1)  # [B, 3]
        vis_pred  = torch.sigmoid(self.vis_head(reg_in)).squeeze(1) * 9.0 + 1.0

        return cls_logits, vis_pred, cos_sim
