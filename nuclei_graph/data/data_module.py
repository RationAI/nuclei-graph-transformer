from collections.abc import Iterable
from typing import Any

import pandas as pd
from hydra.utils import instantiate
from lightning import LightningDataModule
from lightning.pytorch.utilities import rank_zero_info
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

from nuclei_graph.data.datasets.nuclei_dataset import NucleiDataset
from nuclei_graph.data.utils.collator import collate_fn, collate_fn_predict
from nuclei_graph.data.utils.compute_stats import compute_scale_mean
from nuclei_graph.data.utils.sampler import (
    compute_slides_positivity,
    min_count_filter,
)
from nuclei_graph.data.utils.splitter import get_subset, train_val_split
from nuclei_graph.nuclei_graph_typing import Batch, PredictBatch


SUPERVISION_MODES = {"annotation", "cam", "agreement", "agreement-strict"}

BASE_METADATA_COLS = [
    "slide_id",
    "is_carcinoma",
    "slide_nuclei_path",
]

TRAIN_METADATA_COLS = [*BASE_METADATA_COLS, "patient_id", "nuclei_count"]


class DataModule(LightningDataModule):
    def __init__(
        self,
        supervision_mode: str,
        batch_size: int,
        num_workers: int = 0,
        sampler: DictConfig | None = None,
        **datasets: DictConfig,
    ) -> None:
        """Lightning DataModule for nuclei point cloud datasets with weak supervision.

        Args:
            supervision_mode: One of ["annotation", "cam", "agreement", "agreement-strict"].
            batch_size: Number of samples per batch.
            num_workers: Number of DataLoader workers.
            sampler: Optional DictConfig for the sampler to use during training.
            datasets: DictConfigs for datasets for each stage (fit, val, test, predict).

        Supervision Modes Summary:
        ---------------------------------------------------------------------------------------------------------
        Mode              | Mask Logic
        ---------------------------------------------------------------------------------------------------------
        annotation        | All nuclei are supervised, the label is defined only by the annotation ROI.
        cam               | Only confident CAM-labeled nuclei are supervised; uncertain (-1) ignored.
        agreement         | Only nuclei where annotation == CAM are supervised; uncertain CAM (-1) ignored.
        agreement-strict  | Only nuclei positive in both annotation ROI and CAM are supervised; ignore the rest.
        ---------------------------------------------------------------------------------------------------------
        Negative slides supervise all nuclei as negative in all modes.

        The choice of supervision mode only affects positive slides during the training. For validation, testing,
        and prediction, the "agreement-strict" mode is used by default.
        """
        super().__init__()
        assert supervision_mode in SUPERVISION_MODES

        self.supervision_mode = supervision_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampler_partial = sampler
        self.datasets = datasets
        self.positivity: dict[str, float] = {}

        rank_zero_info(
            f"[INFO] Initializing DataModule in the '{self.supervision_mode}' supervision mode."
        )

    def prepare_data(self) -> None:
        uris = {
            uri
            for conf in self.datasets.values()
            if isinstance(conf, DictConfig) and conf.get("uris") is not None
            for uri in conf.uris.values()
            if uri is not None
        }
        for uri in uris:
            download_artifacts(uri)  # download to local cache

    def _instantiate_dataset(self, conf: DictConfig, **kwargs: Any) -> NucleiDataset:
        conf = conf.copy()
        with open_dict(conf):
            conf.pop("uris", None)
        return instantiate(conf, **kwargs)

    def _get_supervision_labels(
        self, conf: DictConfig
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        def load_df(uri_key):
            uri = conf.uris.get(uri_key)
            return pd.read_parquet(download_artifacts(uri)) if uri else None

        df_annot_labels = load_df("annot_labels_uri")
        df_cam_labels = load_df("cam_labels_uri")

        match self.supervision_mode:
            case "annotation":
                assert df_annot_labels is not None
                return df_annot_labels, None
            case "cam":
                assert df_cam_labels is not None
                return None, df_cam_labels
            case "agreement" | "agreement-strict":
                assert df_annot_labels is not None and df_cam_labels is not None
                return df_annot_labels, df_cam_labels
            case _:
                raise ValueError(f"Invalid supervision mode: {self.supervision_mode}")

    def setup(self, stage: str) -> None:
        mode = "train" if stage in ["fit", "validate"] else stage
        conf = self.datasets[mode]
        df_annot_labels, df_cam_labels = self._get_supervision_labels(conf)

        match stage:
            case "fit" | "validate":
                metadata = pd.read_parquet(
                    download_artifacts(conf.uris.metadata_uri),
                    columns=TRAIN_METADATA_COLS,
                )
                df_train, df_val = train_val_split(
                    metadata, keep_cols=TRAIN_METADATA_COLS
                )
                df_train = min_count_filter(df_train, conf.crop_size)
                self.positivity = compute_slides_positivity(
                    df_train, self.supervision_mode, df_annot_labels, df_cam_labels
                )
                scale_mean = (
                    conf.scale_mean
                    if conf.get("scale_mean") is not None
                    else compute_scale_mean(df_train, conf.efd_order)
                )

                train_slides_ids = set(df_train["slide_id"])
                self.train = self._instantiate_dataset(
                    conf,
                    df_metadata=df_train,
                    scale_mean=scale_mean,
                    supervision_mode=self.supervision_mode,
                    df_annot_labels=get_subset(train_slides_ids, df_annot_labels),
                    df_cam_labels=get_subset(train_slides_ids, df_cam_labels),
                )
                val_slides_ids = set(df_val["slide_id"])
                self.val = self._instantiate_dataset(
                    conf,
                    df_metadata=df_val,
                    scale_mean=scale_mean,
                    supervision_mode="agreement-strict",
                    df_annot_labels=get_subset(val_slides_ids, df_annot_labels),
                    df_cam_labels=get_subset(val_slides_ids, df_cam_labels),
                    full_slide=True,
                )
            case "test":
                metadata = pd.read_parquet(
                    download_artifacts(conf.uris.metadata_uri),
                    columns=BASE_METADATA_COLS,
                )
                self.test = self._instantiate_dataset(
                    conf,
                    df_metadata=metadata,
                    scale_mean=conf.scale_mean,  # must be provided in the config
                    supervision_mode="agreement-strict",
                    df_annot_labels=df_annot_labels,
                    df_cam_labels=df_cam_labels,
                )
            case "predict":
                metadata = pd.read_parquet(
                    download_artifacts(conf.uris.metadata_uri),
                    columns=BASE_METADATA_COLS,
                )
                self.predict = self._instantiate_dataset(
                    conf,
                    df_metadata=metadata,
                    scale_mean=conf.scale_mean,  # must be provided in the config
                    supervision_mode="agreement-strict",
                    df_annot_labels=df_annot_labels,
                    df_cam_labels=df_cam_labels,
                    predict=True,
                )

    def train_dataloader(self) -> Iterable[Batch]:
        sampler = (
            instantiate(self.sampler_partial, slides_positivity=self.positivity)(
                dataset=self.train
            )
            if self.sampler_partial is not None
            else None
        )
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable[Batch]:
        return DataLoader(
            self.val,
            batch_size=1,
            num_workers=0,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> Iterable[Batch]:
        return DataLoader(
            self.test,
            batch_size=1,
            num_workers=0,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self) -> Iterable[PredictBatch]:
        return DataLoader(
            self.predict,
            batch_size=1,
            num_workers=0,
            collate_fn=collate_fn_predict,
        )
