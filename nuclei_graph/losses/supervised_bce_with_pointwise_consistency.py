from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nuclei_graph.nuclei_graph_typing import CriterionInput, WSLMasks


class SupervisedBCEWithPointwiseConsistency(nn.Module):
    def __init__(self, consistency_weight: float = 0.5, **kwargs: Any) -> None:
        super().__init__()
        self.consistency_weight = consistency_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        criterion_input: CriterionInput,
        targets_sup: Tensor,
        masks: WSLMasks,
        weight_factor: float = 1.0,
    ) -> tuple[Tensor, dict[str, float]]:
        """Computes combined loss from the supervised BCE and consistency loss between original and augmented views.

        The loss consists of two parts:
          1. supervised loss: BCE computed only on nuclei marked by `masks["sup_mask"]`,
          2. consistency loss: MSE between probabilities of the original and augmented views;
                computed on nuclei outside `masks["ignore_mask"]` and outside masks["sup_mask"].

        It is assumed that training batches do not contain padding.

        Args:
            criterion_input: Dictionary containing model outputs with keys:
                - "logits": Logits from the original input.
                - "logits_aug": (Optional) Logits from an augmented view of the same input.
            targets_sup: Target labels; only for the supervised (confidently labeled) set of nuclei.
            masks: Dictionary of boolean masks with keys:
                - "sup_mask": Selects nuclei for supervised loss.
                - "ignore_mask": Selects nuclei to exclude from all losses.
            weight_factor: Weight factor to scale the consistency loss.

        Returns:
            total_loss: Combined loss tensor.
            logs: Dictionary containing detached loss components and the size of the supervised set.
        """
        logits = criterion_input["logits"]
        logits_sup = logits[masks["sup_mask"]]
        sup_size = targets_sup.numel()

        loss_sup = (
            self.bce(logits_sup, targets_sup) if sup_size > 0 else logits.sum() * 0.0
        )
        loss_cons = logits.sum() * 0.0

        logits_aug = criterion_input.get("logits_aug")
        uncertain_mask = (~masks["ignore_mask"]) & (~masks["sup_mask"])

        if logits_aug is not None and uncertain_mask.any():
            loss_cons = F.mse_loss(
                torch.sigmoid(logits[uncertain_mask]),
                torch.sigmoid(logits_aug[uncertain_mask]),
            )

        total_loss = loss_sup + (self.consistency_weight * weight_factor * loss_cons)

        logs = {
            "loss_sup": loss_sup.detach().item(),
            "loss_cons": loss_cons.detach().item(),
            "sup_size": sup_size,
        }
        return total_loss, logs
