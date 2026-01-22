import torch
import torch.nn as nn
from torch import Tensor


class SupervisedBCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self, logits: Tensor, targets_sup: Tensor, sup_mask: Tensor, ignore_mask: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        """Computes BCE loss on confident nuclei only (e.g., CAM-based supervision).

        Ignores uncertain nuclei completely (hard masking).
        """
        logits_sup = logits[sup_mask]
        sup_size = targets_sup.numel()
        loss = (
            self.bce(logits_sup, targets_sup)
            if sup_size > 0
            else torch.tensor(0.0, device=logits.device, requires_grad=True)
        )
        return loss, {"sup_size": sup_size}
