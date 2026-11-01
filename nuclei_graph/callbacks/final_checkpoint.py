from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint


class FinalModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that also saves `last.ckpt` when training ends.

    With `save_last=True`, Lightning only rewrites `last.ckpt` in epochs that save a
    top-k checkpoint, so it goes stale once the monitored metric stops improving. This
    saves the final weights of a fixed epoch budget (the EMA ones under
    `EMAWeightAveraging`).
    """

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_train_end(trainer, pl_module)
        if trainer.fast_dev_run or not self.save_last:
            return
        self._save_last_checkpoint(trainer, self._monitor_candidates(trainer))
