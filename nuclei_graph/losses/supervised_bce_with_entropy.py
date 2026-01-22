import torch
import torch.nn as nn
from torch import Tensor


class SupervisedBCEWithEntropy(nn.Module):
    def __init__(self, entropy_weight: float = 0.1):
        super().__init__()
        self.entropy_weight = entropy_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self, logits: Tensor, targets: Tensor, sup_mask: Tensor, ignore_mask: Tensor
    ) -> tuple[Tensor, dict[str, float]]:
        """Computes total loss as a combination: BCE(Supervised) + entropy_weight * Mean(Entropy(Uncertain)).

        Nuclei marked by `ignore_mask=True` are excluded from all losses.

        Args:
            logits: Raw model outputs.
            targets: Target labels (e.g., rough annotations).
            sup_mask: Boolean mask; True for high-confidence labels (e.g., CAM-based supervision).
            ignore_mask: Boolean mask; True for regions to ignore (e.g., nuclei in positive slides outside annotations).

        Returns:
            total_loss: Combined loss tensor.
            logs: Dictionary containing detached scalar components.
        """
        logits_sup = logits[sup_mask]
        targets_sup = targets[sup_mask]

        sup_size = targets_sup.numel()

        # it is assumed training batches do not contain padding
        loss_sup = (
            self.bce(logits_sup, targets_sup)
            if sup_size > 0
            else torch.tensor(0.0, device=logits.device, requires_grad=True)
        )

        uncertain_mask = (~ignore_mask) & (~sup_mask)
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
