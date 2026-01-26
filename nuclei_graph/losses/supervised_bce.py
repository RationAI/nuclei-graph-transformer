from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from nuclei_graph.nuclei_graph_typing import CriterionInput, WSLMasks


class SupervisedBCE(nn.Module):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        criterion_input: CriterionInput,
        targets_sup: Tensor,
        masks: WSLMasks,
        **kwargs: Any,
    ) -> tuple[Tensor, dict[str, float]]:
        """Computes BCE loss on confident nuclei only.

        Nuclei outside the supervision mask are ignored (hard masking).

        Args:
            criterion_input: Dictionary with model outputs (contains the key "logits").
            targets_sup: Target labels; only for the supervised (confidently labeled) set of nuclei.
            masks: Dictionary containing boolean mask that selects nuclei for supervised loss ("sup_mask").
            kwargs: Additional keyword arguments.

        Returns:
            loss: Computed BCE loss tensor.
            logs: Dictionary containing the size of the supervised set.
        """
        logits = criterion_input["logits"]
        logits_sup = logits[masks["sup_mask"]]

        sup_size = targets_sup.numel()
        loss_sup = (
            self.bce(logits_sup, targets_sup)
            if sup_size > 0
            else torch.tensor(0.0, device=logits.device, requires_grad=True)
        )
        return loss_sup, {"sup_size": sup_size}
