from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import torch
from hydra.utils import instantiate
from lightning import LightningDataModule
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig, open_dict
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from nuclei_graph.data.block_mask import batch_block_masks
from nuclei_graph.data.datasets.nuclei_dataset import NucleiDataset
from nuclei_graph.nuclei_graph_typing import (
    Batch,
    PartialConf,
    PredictBatch,
    PredictInput,
    Sample,
)


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

    def _instantiate_dataset(self, conf: DictConfig, **kwargs) -> NucleiDataset:
        conf = conf.copy()

        # remove URIs from the config before instantiation
        with open_dict(conf):
            uri_keys = [key for key in conf if str(key).endswith("_uri")]
            for uri_key in uri_keys:
                conf.pop(uri_key, None)

        return instantiate(conf, **kwargs)

    def _train_val_split(
        self, df_metadata: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split metadata into train/validation sets at the patient level."""
        patient_labels = df_metadata.groupby("patient_id")["is_carcinoma"].max()

        train_patients, val_patients = train_test_split(
            patient_labels.index.to_list(),
            test_size=0.1,
            random_state=42,
            stratify=patient_labels.to_list(),
        )
        df_train = df_metadata[df_metadata["patient_id"].isin(train_patients)][
            ["slide_id", "slide_nuclei_path"]
        ]
        df_val = df_metadata[df_metadata["patient_id"].isin(val_patients)][
            ["slide_id", "slide_nuclei_path"]
        ]
        return df_train.reset_index(drop=True), df_val.reset_index(drop=True)

    def compute_slides_positivity(
        self,
        df_metadata: pd.DataFrame,
        df_labels: pd.DataFrame,
        df_cam_indicators: pd.DataFrame | None,
    ) -> dict[int, float]:
        """Computes the fraction of *annotated* positive nuclei per slide."""
        if df_cam_indicators is not None:
            merged = df_labels.merge(df_cam_indicators, on="id", how="inner")
            merged["pos_score"] = (
                merged["label"] * merged["cam_label_indicator"]
            ).astype("uint8")
            positivity_series = merged.groupby("slide_id")["pos_score"].mean()

        else:
            positivity_series = df_labels.groupby("slide_id")["label"].mean()

        return df_metadata["slide_id"].map(positivity_series).fillna(0.0).to_dict()

    def setup(self, stage: str) -> None:
        mode = "train" if stage in ["fit", "validate"] else stage
        conf = self.datasets[mode]
        df_labels = pd.read_parquet(download_artifacts(conf.labels_uri))
        df_cam_indicators = (
            pd.read_parquet(download_artifacts(conf.cam_indicators_uri))
            if conf.get("cam_indicators_uri") is not None
            else None
        )

        match stage:
            case "fit" | "validate":
                df_train, df_val = self._train_val_split(
                    pd.read_csv(Path(download_artifacts(conf.metadata_uri)))
                )

                # filter out slides with insufficient nuclei for cropping during training
                counts = df_train["slide_nuclei_path"].apply(
                    lambda path: pd.read_parquet(path, columns=[]).shape[0]
                )
                df_train = df_train[counts >= conf.crop_size].reset_index(drop=True)

                train_ids = set(df_train["slide_id"])
                self.train = self._instantiate_dataset(
                    conf,
                    df_metadata=df_train,
                    df_labels=df_labels[df_labels["slide_id"].isin(train_ids)],
                    df_cam_indicators=df_cam_indicators[
                        df_cam_indicators["slide_id"].isin(train_ids)
                    ]
                    if df_cam_indicators is not None
                    else None,
                    slides_positivity=self.compute_slides_positivity(
                        df_metadata=df_train,
                        df_labels=df_labels,
                        df_cam_indicators=df_cam_indicators,
                    ),
                )
                val_ids = set(df_val["slide_id"])
                self.val = self._instantiate_dataset(
                    conf,
                    df_metadata=df_val,
                    df_labels=df_labels[df_labels["slide_id"].isin(val_ids)],
                    df_cam_indicators=df_cam_indicators[
                        df_cam_indicators["slide_id"].isin(val_ids)
                    ]
                    if df_cam_indicators is not None
                    else None,
                    full_slide=True,
                )

            case "test":
                self.test = self._instantiate_dataset(
                    conf,
                    df_metadata=pd.read_csv(
                        Path(download_artifacts(conf.metadata_uri)),
                        usecols=["slide_id", "slide_nuclei_path"],
                    ),
                    df_labels=df_labels,
                    df_cam_indicators=df_cam_indicators,
                )

            case "predict":
                self.predict = self._instantiate_dataset(
                    conf,
                    df_metadata=pd.read_csv(
                        Path(download_artifacts(conf.metadata_uri)),
                        usecols=["slide_id", "slide_path", "slide_nuclei_path"],
                    ),
                    df_labels=df_labels,
                    df_cam_indicators=df_cam_indicators,
                )

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
