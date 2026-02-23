from collections.abc import Iterable

import pandas as pd
from hydra.utils import instantiate
from lightning import LightningDataModule
from mlflow.artifacts import download_artifacts
from nuclei_graph.data.supervision import SupervisionStrategy, build_supervision
from omegaconf import DictConfig
from pandas import DataFrame
from torch.utils.data import DataLoader

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
        assert "dataset" in data_params
        assert "mlflow_uris" in data_params

        self.batch_size = batch_size
        self.sup_strategy = instantiate(supervision_strategy)
        self.num_workers = num_workers
        self.sampler_partial = sampler
        self.dataset_conf = data_params["dataset"]
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

    def setup(self, stage: str) -> None:
        mode = "train" if stage in {"fit", "validate"} else stage

        metadata_uri = self.uris_cfg.metadata[mode]

        sup_conf = self.uris_cfg.supervision
        annot_labels = self._load_df(sup_conf.annotation)

        efds_path = self.paths_cfg.features[mode]

        cam_thr_type = self.sup_strategy.cam_thr_type
        eval_sup_strategy = SupervisionStrategy(
            "positive-agreement", "annot_restricted_thr"
        )

        match stage:
            case "fit" | "validate":
                metadata = self._load_df(metadata_uri, cols=TRAIN_METADATA_COLS)

                # --- split train/val ---
                train, val = train_val_split(metadata)
                train = min_count_filter(train, self.dataset_conf.crop_size)
                train_ids = set(train["slide_id"])
                val_ids = set(val["slide_id"])

                # --- load supervision ---
                cam_uri_train = sup_conf.cam[cam_thr_type]
                cam_labels_train = self._load_df(cam_uri_train).pipe(
                    get_subset, train_ids
                )
                annot_labels_train = annot_labels.pipe(get_subset, train_ids)
                sup_train = build_supervision(
                    self.sup_strategy,
                    annot_labels_train,
                    cam_labels_train,
                    self._get_slide_labels(train),
                )

                cam_uri_val = sup_conf.cam.annot_restricted_thr
                cam_labels_val = self._load_df(cam_uri_val).pipe(get_subset, val_ids)
                annot_labels_val = annot_labels.pipe(get_subset, val_ids)
                labels_val = self._get_slide_labels(val)
                sup_val = build_supervision(
                    eval_sup_strategy, annot_labels_val, cam_labels_val, labels_val
                )

                # --- compute statistics for sampler and normalization ---
                self.positivity = {
                    slide_id: slide_sup.nuclei_supervision.get_positivity()
                    for slide_id, slide_sup in sup_train.supervision_map.items()
                }
                log_scale_stats, efd_stats = compute_feature_statistics(
                    train, efds_path, self.dataset_conf.efd_order * 4
                )
                # --- instantiate datasets ---
                self.train = instantiate(
                    self.dataset_conf,
                    metadata=train,
                    log_scale_stats=log_scale_stats,
                    efd_stats=efd_stats,
                    supervision=sup_train,
                    efds_path=efds_path,
                )
                self.val = instantiate(
                    self.dataset_conf,
                    metadata=val,
                    log_scale_stats=log_scale_stats,
                    efd_stats=efd_stats,
                    supervision=sup_val,
                    efds_path=efds_path,
                    full_slide=True,
                )

            case "test":
                metadata = self._load_df(metadata_uri, cols=BASE_METADATA_COLS)
                slide_labels = self._get_slide_labels(metadata)
                cam_labels = self._load_df(sup_conf.cam.annot_restricted_thr)
                sup = build_supervision(
                    eval_sup_strategy, annot_labels, cam_labels, slide_labels
                )

                self.test = instantiate(
                    self.dataset_conf,
                    metadata=metadata,
                    supervision=sup,
                    log_scale_stats=self.dataset_conf.log_scale_stats,
                    efd_stats=self.dataset_conf.efd_stats,
                    efds_path=efds_path,
                    full_slide=True,
                )

            case "predict":
                metadata = self._load_df(metadata_uri, cols=BASE_METADATA_COLS)
                slide_labels = self._get_slide_labels(metadata)
                cam_labels = self._load_df(sup_conf.cam.annot_restricted_thr)
                sup = build_supervision(
                    eval_sup_strategy, annot_labels, cam_labels, slide_labels
                )

                self.predict = instantiate(
                    self.dataset_conf,
                    metadata=metadata,
                    supervision=sup,
                    log_scale_stats=self.dataset_conf.log_scale_stats,
                    efd_stats=self.dataset_conf.efd_stats,
                    efds_path=efds_path,
                    full_slide=True,
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
