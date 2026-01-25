import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nuclei_graph.nuclei_graph_typing import CriterionInput, WSLMasks


class WSLConsistencyLoss(nn.Module):
    def __init__(self, consistency_weight: float = 0.5):
        super().__init__()
        self.consistency_weight = consistency_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self, criterion_input: CriterionInput, targets_sup: Tensor, masks: WSLMasks
    ) -> tuple[Tensor, dict[str, float]]:
        """Computes BCE on high-confidence labels + Consistency on all nuclei marked as `ignore_mask==False`.

        Args:
            criterion_input: Dictionary containing logits from the standard and augmented views.
            targets_sup: Target labels, only for the supervised (confidently labeled) set of nuclei.
            masks: Dictionary containing boolean masks for weakly supervised learning.
        """
        logits_sup = criterion_input["logits"][masks["sup_mask"]]
        sup_size = targets_sup.numel()

        # it is assumed training batches do not contain padding
        loss_sup = (
            self.bce(logits_sup, targets_sup)
            if sup_size > 0
            else torch.tensor(
                0.0, device=criterion_input["logits"].device, requires_grad=True
            )
        )
        preds_orig = torch.sigmoid(criterion_input["logits"][~masks["ignore_mask"]])
        assert criterion_input["logits_aug"] is not None
        preds_aug = torch.sigmoid(criterion_input["logits_aug"][~masks["ignore_mask"]])
        loss_consist = F.mse_loss(preds_orig, preds_aug)
        total_loss = loss_sup + (self.consistency_weight * loss_consist)

        logs = {
            "loss_sup": loss_sup.detach() if isinstance(loss_sup, Tensor) else 0.0,
            "loss_consist": loss_consist.detach(),
            "sup_size": sup_size,
        }

        return total_loss, logs
