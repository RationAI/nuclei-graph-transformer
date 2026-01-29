from typing import Any

import torch.nn as nn
from torch import Tensor

from nuclei_graph.nuclei_graph_typing import WSLMasks


class SupervisedBCE(nn.Module):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        logits: Tensor,
        targets_sup: Tensor,
        wsl_masks: WSLMasks,
        **kwargs: Any,
    ) -> tuple[Tensor, dict[str, float]]:
        """Computes BCE loss on confident nuclei only.

        Nuclei outside the supervision mask are ignored (hard masking) and it is assumed that
        training batches do not contain padding.

        Args:
            logits: Logits from the model, shape (b, n, 1).
            targets_sup: Target labels; only for the supervised set of nuclei, shape (num_supervised, ).
            wsl_masks: Dictionary containing mask that selects nuclei for supervised loss ("sup_mask" (tensor[bool])), shape (b, n, ).
            kwargs: Additional keyword arguments.

        Returns:
            loss: Computed BCE loss tensor.
            logs: Dictionary containing the size of the supervised set.
        """
        logits_sup = logits.squeeze(-1)[wsl_masks["sup_mask"]]
        sup_size = targets_sup.numel()
        loss_sup = (
            self.bce(logits_sup, targets_sup) if sup_size > 0 else logits.sum() * 0.0
        )
        return loss_sup, {"sup_size": sup_size}
