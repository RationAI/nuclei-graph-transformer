import torch
import torch.nn as nn
from torch import Tensor

from nuclei_graph.nuclei_graph_typing import CriterionInput, WSLMasks


class SupervisedBCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self, criterion_input: CriterionInput, targets_sup: Tensor, masks: WSLMasks
    ) -> tuple[Tensor, dict[str, float]]:
        """Computes BCE loss on confident nuclei only (e.g., CAM-based supervision).

        Ignores uncertain nuclei (outside the supervision mask) completely (hard masking).

        Args:
            criterion_input: Dictionary containing model outputs.
            targets_sup: Target labels, only for the supervised (confidently labeled) set of nuclei.
            masks: Dictionary containing boolean masks for weakly supervised learning.
        """
        logits = criterion_input["logits"]
        logits_sup = logits[masks["sup_mask"]]

        sup_size = targets_sup.numel()
        loss = (
            self.bce(logits_sup, targets_sup)
            if sup_size > 0
            else torch.tensor(0.0, device=logits.device, requires_grad=True)
        )
        return loss, {"sup_size": sup_size}
