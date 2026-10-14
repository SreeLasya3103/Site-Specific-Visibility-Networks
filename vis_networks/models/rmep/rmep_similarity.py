import torch
import torch.nn as nn
import torch.nn.functional as F
from .rmep import Model as RMEPModel

# RMEP takes the plain image, so there is no custom transform (as in rmep.py).
CUSTOM_TRANSFORM = None

PROJ_DIM = 128   # same embedding size as VisNet-Similarity


class Model(nn.Module): #VisNet-Similarity with the frozen pooled RMEP backbone in place of VisNet's. Everything after the backbone (projector, heads, loss) is the same as in
#models/visnet/visnet_similarity.py, so the two differ only in the backbone.
    
    def __init__(self, n_classes, input_size, mean=None, std=None,
                 dropout_p=0.4, pretrained_path=None, **kwargs):
        super().__init__()
        # input_size is [2, 3, H, W]; base RMEP expects [3, H, W]
        single_size = list(input_size[1:])

        if not pretrained_path:
            raise ValueError(
                "RMEP-Similarity requires model_params['pretrained_path'] (the pooled RMEP "
                "checkpoint). A frozen randomly-initialized backbone carries no learned features. "
                f"Received kwargs: {list(kwargs.keys())}"
            )

        # mean/std are not passed on: they come from the checkpoint, so the frozen
        # backbone sees inputs normalized exactly as during its own training.
        base = RMEPModel(n_classes, single_size)
        # Trigger LazyLinear initialization so the checkpoint's last layer can load
        with torch.no_grad():
            base(torch.zeros(1, *single_size))

        sd = torch.load(pretrained_path, weights_only=True, map_location='cpu')
        report = base.load_state_dict(sd, strict=False)
        # strict=False is only there because checkpoints also carry example_input
        # and example_output. Anything missing means the backbone did not load
        # and would silently stay random.
        if report.missing_keys:
            raise RuntimeError(f"Backbone checkpoint {pretrained_path} is missing "
                               f"{len(report.missing_keys)} tensors: {report.missing_keys}")
        extra = [k for k in report.unexpected_keys
                 if k not in ('example_input', 'example_output')]
        if extra:
            raise RuntimeError(f"Backbone checkpoint {pretrained_path} has tensors an RMEP "
                               f"backbone does not have (wrong checkpoint?): {extra}")
        print(f"RMEP-Similarity: loaded {len(base.state_dict())} backbone tensors "
              f"from {pretrained_path}")

        # Normalization buffers from the checkpoint (shaped [1, 3, 1, 1])
        self.register_buffer('mean', base.mean.clone())
        self.register_buffer('std',  base.std.clone())

        # Frozen backbone: every RMEP layer except the final classifier (LazyLinear),
        # ending in the flattened 256-dim feature vector the LDA in Chapter 7 uses.
        self.backbone = nn.Sequential(*list(base.model.children())[:-1])
        del base
        for p in self.backbone.parameters():
            p.requires_grad = False

        with torch.no_grad():
            feat_dim = self.backbone(torch.zeros(1, *single_size)).size(1)

        # Shared projector: backbone features -> 128-dim embedding
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, PROJ_DIM),
            nn.ReLU(True),
        )

        # Classification head: cat([p_curr, p_diff]) -> n_classes
        self.cls_head = nn.Sequential(
            nn.Linear(PROJ_DIM * 2, 64),
            nn.ReLU(True),
            nn.Dropout(dropout_p),
            nn.Linear(64, n_classes),
        )

        # Regression head: [cos_sim, ||f_diff||, ||f_curr||] -> visibility (1-10 mi)
        self.vis_head = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(True),
            nn.Linear(32, 1),
        )

    def train(self, mode=True):
        # RMEP's residual blocks contain Dropout. Keep the frozen backbone in eval
        # mode whatever mode the rest of the model is in, so the current and
        # reference images never get independent dropout masks.
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, x):
        # x: [B, 2, 3, H, W]  (current @ dim-1 index 0, reference @ index 1)
        current   = x[:, 0].subtract(self.mean).divide(self.std)
        reference = x[:, 1].subtract(self.mean).divide(self.std)

        # Frozen backbone: no gradient storage needed
        with torch.no_grad():
            f_curr = self.backbone(current)     # [B, 256]
            f_ref  = self.backbone(reference)   # [B, 256]

        f_diff = torch.abs(f_curr - f_ref)

        # Project into shared embedding space
        p_curr = self.projector(f_curr)    # [B, 128]
        p_ref  = self.projector(f_ref)     # [B, 128]
        p_diff = self.projector(f_diff)    # [B, 128]

        # Cosine similarity between projected current and reference embeddings
        cos_sim = F.cosine_similarity(p_curr, p_ref, dim=1)   # [B]

        # Classification: concat current + diff projections
        cls_logits = self.cls_head(torch.cat([p_curr, p_diff], dim=1))   # [B, n_classes]

        # Regression: similarity, absolute diff magnitude, current magnitude
        diff_norm_n = torch.log1p(torch.norm(f_diff, dim=1, keepdim=True))
        curr_norm_n = torch.log1p(torch.norm(f_curr, dim=1, keepdim=True))
        reg_in   = torch.cat([cos_sim.unsqueeze(1), diff_norm_n, curr_norm_n], dim=1)  # [B, 3]
        vis_pred = torch.sigmoid(self.vis_head(reg_in)).squeeze(1) * 9.0 + 1.0

        return cls_logits, vis_pred, cos_sim
