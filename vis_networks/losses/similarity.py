import torch
import torch.nn as nn


class SimilarityVisibilityLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1, gamma=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.register_buffer('centers',
                             torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0,
                                           6.0, 7.0, 8.0, 9.0, 10.0]))

    def forward(self, output, labels_onehot):
        cls_logits, vis_pred, cos_sim = output
        class_idx = labels_onehot.argmax(dim=1)
        centers = self.centers.to(cls_logits.device)
        true_vis = centers[class_idx]

        loss_cls = self.ce(cls_logits, class_idx)
        loss_reg = self.mse(vis_pred, true_vis)
        true_vis_norm = (true_vis - 1.0) / 9.0
        loss_corr = self.mse(cos_sim, true_vis_norm)

        return self.alpha * loss_cls + self.beta * loss_reg + self.gamma * loss_corr
