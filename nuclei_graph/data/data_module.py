from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
from hydra.utils import instantiate
from lightning import LightningDataModule
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from nuclei_graph.data.block_mask import batch_block_masks
from nuclei_graph.nuclei_graph_typing import (
    Batch,
    PartialConf,
    PredictBatch,
    PredictInput,
    Sample,
)
from preprocessing.slide_helpers import get_ground_truth


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

    def _train_val_split(
        self, df_metadata: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_metadata["label"] = df_metadata["slide_mrxs_path"].apply(get_ground_truth)

        patient_labels = df_metadata.groupby("patient_id")["label"].max()
        unique_patients = patient_labels.index.to_list()
        patient_y = patient_labels.to_list()

        train_patients, val_patients = train_test_split(
            unique_patients,
            test_size=0.1,
            random_state=42,
            stratify=patient_y,
        )
        df_metadata_train = df_metadata[
            df_metadata["patient_id"].isin(train_patients)
        ].drop(columns=["patient_id", "label"])
        df_metadata_val = df_metadata[
            df_metadata["patient_id"].isin(val_patients)
        ].drop(columns=["patient_id", "label"])

        return df_metadata_train.reset_index(drop=True), df_metadata_val.reset_index(
            drop=True
        )

    def _check_nuclei_count(self, df_metadata: pd.DataFrame) -> pd.DataFrame:
        """Filters out slides with insufficient nuclei for cropping during training."""
        df_tmp = df_metadata.copy()
        df_tmp["nuclei_count"] = df_tmp["slide_nuclei_path"].apply(
            lambda path: len(pd.read_parquet(path))
        )
        df_tmp = df_tmp[df_tmp["nuclei_count"] >= self.datasets["train"].crop_size]
        return df_tmp.reset_index(drop=True)

    def _prepare_split(self, conf: DictConfig):
        if conf.get("annot_masks_path") is not None:
            conf.annot_masks_path = download_artifacts(conf.annot_masks_path)

        df_metadata = pd.read_csv(Path(download_artifacts(conf.metadata_path)))
        df_metadata_train, df_metadata_val = self._train_val_split(df_metadata)
        df_metadata_train = self._check_nuclei_count(df_metadata_train)

        conf_train, conf_val = deepcopy(conf), deepcopy(conf)
        conf_train.df_metadata = df_metadata_train
        conf_val.df_metadata = df_metadata_val
        conf_train.pop("metadata_path", None)
        conf_val.pop("metadata_path", None)

        return instantiate(conf_train), instantiate(conf_val)

    def _prepare_single(self, conf: DictConfig):
        conf_copy = deepcopy(conf)

        if conf.get("annot_masks_path") is not None:
            conf_copy.annot_masks_path = download_artifacts(conf.annot_masks_path)

        conf_copy.df_metadata = pd.read_csv(
            Path(download_artifacts(conf.metadata_path))
        )
        conf_copy.pop("metadata_path", None)
        return instantiate(conf_copy)

    def setup(self, stage: str) -> None:
        match stage:
            case "fit" | "validate":
                self.train, self.val = self._prepare_split(self.datasets["train"])
            case "test":
                self.test = self._prepare_single(self.datasets["test"])
            case "predict":
                self.predict = self._prepare_single(self.datasets["predict"])

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
