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
CAM_THRESHOLD_TYPES = {"annot_restricted_thr", "default_thr"}

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
        cam_thr_type: str,
        batch_size: int,
        num_workers: int = 0,
        sampler: DictConfig | None = None,
        **datasets: DictConfig,
    ) -> None:
        """Lightning DataModule for nuclei point cloud datasets with weak supervision.

        Args:
            supervision_mode (str): Supervision mode to use for positive slides. One of "annotation", "cam", "agreement", "agreement-strict".
            cam_thr_type (str): CAM threshold type to use for positive slides. One of "annot_restricted_thr", "default_thr":
                "annot_restricted_thr" — should be paired with a supervision mode that restricts to annotated regions.
                "default_thr" — more strict (higher), doesn't assume any restrictions.
            batch_size (int): Batch size for training.
            num_workers (int, optional): Number of workers for data loading. Defaults to 0.
            sampler (DictConfig | None, optional): Sampler configuration for training data loader. Defaults to None.
            **datasets (DictConfig): Dataset configurations for different stages.

        Supervision Modes Summary:
        ---------------------------------------------------------------------------------------------------------
        Mode              | Mask Logic
        ---------------------------------------------------------------------------------------------------------
        annotation        | All nuclei are supervised, the label is defined only by the annotation ROI.
        cam               | Only confident CAM-labeled nuclei are supervised (positive/negative); uncertain (-1) ignored.
        agreement         | Only nuclei where annotation == CAM are supervised; uncertain CAM (-1) ignored.
        agreement-strict  | Only positive nuclei inside both annotation ROI and CAM ROI are supervised; ignore the rest.
        ---------------------------------------------------------------------------------------------------------
        Negative slides supervise all nuclei as negative in all modes.
        The choice of supervision mode only affects positive slides during the training (fit stage).

        For validation, testing, and prediction the default is "agreement-strict" supervision mode and "default_thr" CAM threshold type.
        """
        super().__init__()
        assert supervision_mode in SUPERVISION_MODES
        assert cam_thr_type in CAM_THRESHOLD_TYPES

        self.supervision_mode = supervision_mode
        self.cam_thr_type = cam_thr_type
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

    def _load_df(self, uri: str, columns: list[str] | None = None) -> pd.DataFrame:
        return pd.read_parquet(
            download_artifacts(uri),
            columns=columns,
        )

    def setup(self, stage: str) -> None:
        mode = "train" if stage in ["fit", "validate"] else stage
        conf = self.datasets[mode]

        cam_label_uris = conf.uris.cam_label_uris
        df_cam_labels = self._load_df(cam_label_uris.default_thr)
        df_annot_labels = self._load_df(conf.uris.annot_labels_uri)

        match stage:
            case "fit" | "validate":
                metadata = self._load_df(conf.uris.metadata_uri, TRAIN_METADATA_COLS)
                df_train, df_val = train_val_split(
                    metadata, keep_cols=TRAIN_METADATA_COLS
                )
                df_train = min_count_filter(df_train, conf.crop_size)
                df_cam_labels_train = self._load_df(cam_label_uris[self.cam_thr_type])
                self.positivity = compute_slides_positivity(
                    df_train,
                    self.supervision_mode,
                    df_annot_labels,
                    df_cam_labels_train,
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
                    df_annot_labels=get_subset(train_slides_ids, df_annot_labels),
                    df_cam_labels=get_subset(train_slides_ids, df_cam_labels_train),
                    scale_mean=scale_mean,
                    supervision_mode=self.supervision_mode,
                )

                val_slides_ids = set(df_val["slide_id"])
                self.val = self._instantiate_dataset(
                    conf,
                    df_metadata=df_val,
                    df_annot_labels=get_subset(val_slides_ids, df_annot_labels),
                    df_cam_labels=get_subset(val_slides_ids, df_cam_labels),
                    scale_mean=scale_mean,
                    supervision_mode="agreement-strict",
                    full_slide=True,
                )
            case "test":
                metadata = self._load_df(conf.uris.metadata_uri, BASE_METADATA_COLS)
                self.test = self._instantiate_dataset(
                    conf,
                    df_metadata=metadata,
                    df_annot_labels=df_annot_labels,
                    df_cam_labels=df_cam_labels,
                    scale_mean=conf.scale_mean,  # must be provided in the config
                    supervision_mode="agreement-strict",
                )
            case "predict":
                metadata = self._load_df(conf.uris.metadata_uri, BASE_METADATA_COLS)
                self.predict = self._instantiate_dataset(
                    conf,
                    df_metadata=metadata,
                    df_annot_labels=df_annot_labels,
                    df_cam_labels=df_cam_labels,
                    scale_mean=conf.scale_mean,  # must be provided in the config
                    supervision_mode="agreement-strict",
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
