from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from nuclei_graph.nuclei_graph_typing import WSLMasks


class SupervisedBCEWithPointwiseConsistency(nn.Module):
    def __init__(self, consistency_weight: float = 0.5, **kwargs: Any) -> None:
        super().__init__()
        self.consistency_weight = consistency_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: Tensor,
        targets_sup: Tensor,
        wsl_masks: WSLMasks,
        logits_aug: Tensor,
        weight_factor: float = 1.0,
        **kwargs: Any,
    ) -> tuple[Tensor, dict[str, float]]:
        """Computes combined loss from the supervised BCE and consistency loss between original and augmented views.

        The loss consists of two parts:
          1. supervised loss: BCE computed only on nuclei marked by `wsl_masks["sup_mask"]`,
          2. consistency loss: MSE between probabilities of the original and augmented views;
                computed on nuclei outside `wsl_masks["ignore_mask"]` and outside `wsl_masks["sup_mask"]`.

        It is assumed that training batches do not contain padding.

        Args:
            logits: Logits from the model (on the original input), shape (b, n, 1).
            targets_sup: Target labels; only for the supervised set of nuclei, shape (num_supervised, ).
            wsl_masks: Dictionary of masks with keys:
                - "sup_mask" (tensor[bool]): Selects nuclei for supervised loss, shape (b, n, ).
                - "ignore_mask" (tensor[bool]): Selects nuclei to exclude from all losses, shape (b, n, ).
            logits_aug: Logits from the model (on the augmented input), shape (b, n, 1).
            weight_factor: Weight factor to scale the consistency loss.
            kwargs: Additional keyword arguments.

        Returns:
            total_loss: Combined loss tensor.
            logs: Dictionary containing detached loss components and the size of the supervised set.
        """
        logits = logits.squeeze(-1)
        logits_aug = logits_aug.squeeze(-1)
        logits_sup = logits[wsl_masks["sup_mask"]]

        sup_size = targets_sup.numel()

        loss_sup = (
            self.bce(logits_sup, targets_sup) if sup_size > 0 else logits.sum() * 0.0
        )
        loss_cons = logits.sum() * 0.0

        uncertain_mask = (~wsl_masks["ignore_mask"]) & (~wsl_masks["sup_mask"])

        if uncertain_mask.any():
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
