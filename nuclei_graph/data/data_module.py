from collections.abc import Iterable

import torch
from hydra.utils import instantiate
from lightning import LightningDataModule
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from nuclei_graph.data.datasets import NucleiDataset
from nuclei_graph.typing import (
    Batch,
    PartialConf,
    PredictBatch,
    PredictInput,
    Sample,
)
from nuclei_graph.utils import batch_block_masks


class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        sampler: PartialConf | None = None,
        **datasets: DictConfig,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets = datasets
        self.sampler_partial = sampler
        self.batch_block_masks = batch_block_masks

    def setup(self, stage: str) -> None:
        def prepare(conf: DictConfig) -> NucleiDataset:
            conf.metadata_path = download_artifacts(conf.metadata_path)
            conf.nuclei_path = download_artifacts(conf.nuclei_path)
            conf.graphs_path = download_artifacts(conf.graphs_path)
            if conf.get("annot_masks_path") is not None:
                conf.annot_masks_path = download_artifacts(conf.annot_masks_path)
            return instantiate(conf)

        match stage:
            case "fit":
                self.train = prepare(self.datasets["train"])
                self.val = prepare(self.datasets["val"])
            case "validate":
                self.val = prepare(self.datasets["val"])
            case "test":
                self.test = prepare(self.datasets["test"])
            case "predict":
                self.predict = prepare(self.datasets["predict"])

    def _collate_fn(self, batch: Batch) -> Sample:
        return {
            "x": torch.stack([b["x"] for b in batch], dim=0),
            "pos": torch.stack([b["pos"] for b in batch], dim=0),
            "y": torch.cat([b["y"] for b in batch], dim=0),  # variable-length tensors
            "annot_mask": torch.stack([b["annot_mask"] for b in batch], dim=0),
            "block_mask": self.batch_block_masks([b["block_mask"] for b in batch]),
        }

    def _collate_fn_predict(self, batch: PredictBatch) -> PredictInput:
        items, metadata = zip(*batch, strict=True)
        return self._collate_fn(list(items)), list(metadata)

    def train_dataloader(self) -> Iterable[Sample]:
        sampler = (
            instantiate(
                self.sampler_partial, slides_positivity=self.train.slides_positivity
            )(dataset=self.train)
            if self.sampler_partial is not None
            else None
        )
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            collate_fn=self._collate_fn,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable[Sample]:
        return DataLoader(
            self.val,
            batch_size=1,  # process full graphs
            num_workers=0,
            collate_fn=self._collate_fn,
        )

    def test_dataloader(self) -> Iterable[Sample]:
        return DataLoader(
            self.test,
            batch_size=1,  # process full graphs
            num_workers=0,
            collate_fn=self._collate_fn,
        )

    def predict_dataloader(self) -> Iterable[PredictInput]:
        return DataLoader(
            self.predict,
            batch_size=1,  # process full graphs
            num_workers=0,
            collate_fn=self._collate_fn_predict,
        )
