from collections.abc import Iterable

import pandas as pd
import torch
from hydra.utils import instantiate
from lightning import LightningDataModule
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import DataLoader

from nuclei_graph.data.supervision import (
    DatasetSupervision,
    SupervisionStrategy,
    build_supervision,
)
from nuclei_graph.data.utils import (
    collate_fn,
    collate_fn_predict,
    compute_feature_statistics,
    get_subset,
    min_count_filter,
    train_val_split,
)
from nuclei_graph.nuclei_graph_typing import (
    Batch,
    PredictBatch,
)


EVAL_CAM_THR_TYPE = "annot_restricted_thr"
EVAL_SUP_STRATEGY = SupervisionStrategy("positive-agreement", EVAL_CAM_THR_TYPE)

BASE_METADATA_COLS = [
    "slide_id",
    "is_carcinoma",
    "slide_nuclei_path",
]

TRAIN_METADATA_COLS = [*BASE_METADATA_COLS, "patient_id", "nuclei_count"]


class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        supervision_strategy: DictConfig,
        num_workers: int = 0,
        sampler: DictConfig | None = None,
        **data_params: DictConfig,
    ) -> None:
        """Lightning DataModule for nuclei point cloud datasets with weak supervision.

        Args:
            batch_size: Batch size for training.
            num_workers: Number of workers for data loading. Defaults to 0.
            supervision_strategy: An DictConfig defining the type of supervision to use for positive slides.
            sampler: Sampler configuration for training data loader. Defaults to None.
            **data_params: Additional parameters expected to contain keys:
                - dataset: DictConfig for instantiation of a Torch Dataset.
                - mlflow_uris: DictConfig with MLflow keys "supervision" and "metadata" containing URIs for respective artifacts.
                - paths: DictConfig with key "features" containing paths to nuclei EFD representations.

        The choice of supervision strategy only affects positive slides during the training (fit stage).
        For validation, testing, and prediction the default is "positive-agreement" supervision strategy and "annot_restricted_thr" CAM threshold type.
        """
        super().__init__()
        self.batch_size = batch_size
        self.sup_strategy = instantiate(supervision_strategy)
        self.num_workers = num_workers
        self.sampler_partial = sampler
        self.dataset_cfg = data_params["dataset"]
        self.uris_cfg = data_params["mlflow_uris"]
        self.paths_cfg = data_params["paths"]
        self.positivity: dict[str, float] = {}

    def _get_slide_labels(
        self, df: DataFrame, label_col: str = "is_carcinoma"
    ) -> dict[str, int]:
        return {str(k): int(v) for k, v in df.set_index("slide_id")[label_col].items()}

    def _load_df(self, uri: str, cols: list[str] | None = None) -> pd.DataFrame:
        path = download_artifacts(uri)
        return pd.read_parquet(path, columns=cols)

    def _prepare_supervision(
        self,
        df: DataFrame,
        strategy: SupervisionStrategy,
        cam_uri: str,
        annot_labels: DataFrame,
    ) -> DatasetSupervision:
        slide_ids = set(df["slide_id"])

        cam_labels = self._load_df(cam_uri).pipe(get_subset, slide_ids)
        annot_labels = annot_labels.pipe(get_subset, slide_ids)

        return build_supervision(
            strategy,
            annot_labels,
            cam_labels,
            self._get_slide_labels(df),
        )

    def _prepare_stats(
        self,
        df: DataFrame,
        efds_path: str,
        target_dim: int,
        log_scales_list: list[float] | None = None,
        efd_stats_cfg: DictConfig | None = None,
    ) -> tuple[tuple[float, ...], dict[str, Tensor]]:

        if log_scales_list is None or efd_stats_cfg is None:
            return compute_feature_statistics(df, efds_path, target_dim)

        log_scales = tuple(log_scales_list)
        efd_stats = {
            "mean": torch.tensor(efd_stats_cfg.mean, dtype=torch.float32)[:target_dim],
            "std": torch.tensor(efd_stats_cfg.std, dtype=torch.float32)[:target_dim],
        }
        return log_scales, efd_stats

    def setup(self, stage: str) -> None:
        mode = "train" if stage in {"fit", "validate"} else stage

        sup_conf = self.uris_cfg.supervision
        annot_labels = self._load_df(sup_conf.annotation)
        metadata_uri = self.uris_cfg.metadata[mode]
        efds_path = self.paths_cfg.features[mode]
        target_dim = self.dataset_cfg.efd_order * 4

        match stage:
            case "fit" | "validate":
                metadata = self._load_df(metadata_uri, cols=TRAIN_METADATA_COLS)

                # --- split train/val ---
                train, val = train_val_split(metadata)
                train = min_count_filter(train, self.dataset_cfg.crop_size)

                # --- load supervision ---
                sup_train = self._prepare_supervision(
                    df=train,
                    strategy=self.sup_strategy,
                    cam_uri=sup_conf.cam[self.sup_strategy.cam_thr_type],
                    annot_labels=annot_labels,
                )
                sup_val = self._prepare_supervision(
                    df=val,
                    strategy=EVAL_SUP_STRATEGY,
                    cam_uri=sup_conf.cam[EVAL_CAM_THR_TYPE],
                    annot_labels=annot_labels,
                )

                # --- compute statistics for sampler and normalization ---
                self.positivity = {
                    slide_id: slide_sup.nuclei_supervision.get_positivity()
                    for slide_id, slide_sup in sup_train.supervision_map.items()
                }
                log_scales, efd_stats = self._prepare_stats(
                    df=train,
                    efds_path=efds_path,
                    target_dim=target_dim,
                    log_scales_list=self.dataset_cfg.get("log_scale_stats", None),
                    efd_stats_cfg=self.dataset_cfg.get("efd_stats", None),
                )

                # --- instantiate datasets ---
                self.train = instantiate(
                    self.dataset_cfg,
                    metadata=train,
                    log_scale_stats=log_scales,
                    efd_stats=efd_stats,
                    supervision=sup_train,
                    efds_path=efds_path,
                )
                self.val = instantiate(
                    self.dataset_cfg,
                    metadata=val,
                    log_scale_stats=log_scales,
                    efd_stats=efd_stats,
                    supervision=sup_val,
                    efds_path=efds_path,
                    full_slide=True,
                )

            case "test" | "predict":
                metadata = self._load_df(metadata_uri, cols=BASE_METADATA_COLS)
                sup = self._prepare_supervision(
                    df=metadata,
                    strategy=EVAL_SUP_STRATEGY,
                    cam_uri=sup_conf.cam[EVAL_CAM_THR_TYPE],
                    annot_labels=annot_labels,
                )
                log_scales, efd_stats = self._prepare_stats(
                    df=metadata,
                    efds_path=efds_path,
                    target_dim=target_dim,
                    log_scales_list=self.dataset_cfg.log_scale_stats,  # must be set
                    efd_stats_cfg=self.dataset_cfg.efd_stats,  # must be set
                )

                dataset = instantiate(
                    self.dataset_cfg,
                    metadata=metadata,
                    supervision=sup,
                    log_scale_stats=log_scales,
                    efd_stats=efd_stats,
                    efds_path=efds_path,
                    full_slide=True,
                    predict=(stage == "predict"),
                )

                if stage == "test":
                    self.test = dataset
                else:
                    self.predict = dataset

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
