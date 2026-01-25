from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from nuclei_graph.nuclei_graph_typing import CriterionInput, WSLMasks


class SupervisedBCEWithEntropy(nn.Module):
    def __init__(self, entropy_weight: float = 0.1, **kwargs: Any) -> None:
        super().__init__()
        self.entropy_weight = entropy_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self, criterion_input: CriterionInput, targets_sup: Tensor, masks: WSLMasks
    ) -> tuple[Tensor, dict[str, float]]:
        """Computes total loss as a combination: BCE(Supervised) + entropy_weight * Mean(Entropy(Uncertain)).

        Nuclei marked by `ignore_mask=True` are excluded from all losses.

        Args:
            criterion_input: Dictionary containing model outputs.
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

        uncertain_mask = (~masks["ignore_mask"]) & (~masks["sup_mask"])
        probs_uncertain = torch.sigmoid(logits[uncertain_mask])

        entropy = -(
            probs_uncertain * torch.log(probs_uncertain + 1e-8)
            + (1 - probs_uncertain) * torch.log(1 - probs_uncertain + 1e-8)
        )
        loss_entropy = (
            entropy.mean()
            if probs_uncertain.numel() > 0
            else torch.tensor(0.0, device=logits.device, requires_grad=True)
        )

        total_loss = loss_sup + self.entropy_weight * loss_entropy

        logs = {
            "loss_sup": loss_sup.detach(),
            "loss_ent": loss_entropy.detach(),
            "sup_size": sup_size,
        }
        return total_loss, logs
