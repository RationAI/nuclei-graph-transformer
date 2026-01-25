from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nuclei_graph.nuclei_graph_typing import CriterionInput, WSLMasks


class SupervisedBCEWithConsistency(nn.Module):
    def __init__(self, consistency_weight: float = 0.5, **kwargs: Any) -> None:
        super().__init__()
        self.consistency_weight = consistency_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self, criterion_input: CriterionInput, targets_sup: Tensor, masks: WSLMasks
    ) -> tuple[Tensor, dict[str, float]]:
        """Computes BCE on high-confidence labels + Consistency on all nuclei marked as False by the provided ignore mask.

        Args:
            criterion_input: Dictionary containing logits from the standard and augmented views.
            targets_sup: Target labels, only for the supervised (confidently labeled) set of nuclei.
            masks: Dictionary containing boolean masks ("sup_mask" and "ignore_mask") for weakly supervised learning.

        Returns:
            total_loss: Combined loss tensor.
            logs: Dictionary containing detached loss components and the size of the supervised set.
        """
        logits = criterion_input["logits"]
        logits_sup = logits[masks["sup_mask"]]
        sup_size = targets_sup.numel()

        # it is assumed training batches do not contain padding
        loss_sup = (
            self.bce(logits_sup, targets_sup)
            if sup_size > 0
            else torch.tensor(0.0, device=logits.device, requires_grad=True)
        )
        loss_consist = torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits_aug = criterion_input.get("logits_aug")
        uncertain_mask = ~masks["ignore_mask"]

        if logits_aug is not None and uncertain_mask.any():
            loss_consist = F.mse_loss(
                torch.sigmoid(logits[uncertain_mask]),
                torch.sigmoid(logits_aug[uncertain_mask]),
            )

        total_loss = loss_sup + (self.consistency_weight * loss_consist)

        logs = {
            "loss_sup": loss_sup.detach() if isinstance(loss_sup, Tensor) else 0.0,
            "loss_consist": loss_consist.detach(),
            "sup_size": sup_size,
        }
        return total_loss, logs
