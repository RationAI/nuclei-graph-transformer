from typing import Any

from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EMAWeightAveraging


class ParamsEMAWeightAveraging(EMAWeightAveraging):
    """EMA of the parameters, with the current model's buffers (BatchNorm statistics).

    Averaging the buffers keeps decay^steps of their initial values, which matters for
    features with ~zero variance (e.g. the constant normalized EFD coefficients):
    BatchNorm divides by sqrt(running_var + eps), so the averaged running_var (~1e-5
    instead of ~0) rescales them and shifts the predictions. Lightning swaps the
    buffers together with the parameters, so they are copied after every update.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(use_buffers=False, **kwargs)

    def _copy_buffers(self, pl_module: LightningModule) -> None:
        assert self._average_model is not None
        for average_buffer, buffer in zip(
            self._average_model.module.buffers(), pl_module.buffers(), strict=True
        ):
            average_buffer.data.copy_(buffer.data)

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        self._copy_buffers(pl_module)
