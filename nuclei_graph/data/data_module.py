from collections.abc import Iterable

import pandas as pd
from hydra.utils import instantiate
from lightning import LightningDataModule
from lightning.pytorch.utilities import rank_zero_info
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from pandas import DataFrame
from torch.utils.data import DataLoader

from nuclei_graph.data.utils import (
    build_supervision,
    collate_fn,
    collate_fn_predict,
    compute_scale_mean,
    compute_slides_positivity,
    get_subset,
    min_count_filter,
    train_val_split,
)
from nuclei_graph.nuclei_graph_typing import (
    Batch,
    PredictBatch,
)


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
        batch_size: int,
        num_workers: int = 0,
        supervision_mode: str | None = "agreement-strict",
        cam_thr_type: str | None = "annot_restricted_thr",
        sampler: DictConfig | None = None,
        **data_params: DictConfig,
    ) -> None:
        """Lightning DataModule for nuclei point cloud datasets with weak supervision.

        Args:
            batch_size: Batch size for training.
            num_workers: Number of workers for data loading. Defaults to 0.
            supervision_mode: Optional supervision mode to use for positive slides. One of "annotation", "cam", "agreement",
                "agreement-strict". If None, defaults to "agreement-strict".
            cam_thr_type: Optional CAM threshold type to use for positive slides. One of "annot_restricted_thr", "default_thr":
                "annot_restricted_thr" — should be paired with a supervision mode that restricts to annotated regions.
                "default_thr" — more strict (higher), doesn't assume any restrictions.
                If None, defaults to "annot_restricted_thr".
            sampler: Sampler configuration for training data loader. Defaults to None.
            **data_params: Additional parameters expected to contain keys:
                - dataset: DictConfig for instantiation of a Torch Dataset.
                - mlflow_uris: DictConfig with MLflow keys "supervision" and "metadata" containing URIs for respective artifacts.
                - paths: DictConfig with key "features" containing paths to nuclei EFD representations.

        Supervision Modes Summary:
        -----------------------------------------------------------------------------------------------------------------
        Mode              | Mask Logic
        -----------------------------------------------------------------------------------------------------------------
        annotation        | All nuclei are supervised, the label is defined only by the annotation ROI.
        cam               | Only confident CAM-labeled nuclei are supervised (positive/negative); uncertain (-1) ignored.
        agreement         | Only nuclei where annotation == CAM are supervised; uncertain CAM (-1) ignored.
        agreement-strict  | Only positive nuclei inside both annotation ROI and CAM ROI are supervised; ignore the rest.
        -----------------------------------------------------------------------------------------------------------------
        Negative slides supervise all nuclei as negative in all modes.
        The choice of supervision mode only affects positive slides during the training (fit stage).

        For validation, testing, and prediction the default is "agreement-strict" supervision mode and "annot_restricted_thr" CAM threshold type.
        """
        super().__init__()
        assert supervision_mode in SUPERVISION_MODES
        assert cam_thr_type in CAM_THRESHOLD_TYPES
        assert "dataset" in data_params
        assert "mlflow_uris" in data_params

        self.supervision_mode = supervision_mode
        self.cam_thr_type = cam_thr_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sampler_partial = sampler
        self.dataset_conf = data_params["dataset"]
        self.uris_cfg = data_params["mlflow_uris"]
        self.paths_cfg = data_params["paths"]
        self.positivity: dict[str, float] = {}

        rank_zero_info(
            f"[INFO] Initializing DataModule in the '{self.supervision_mode}' supervision mode and '{self.cam_thr_type}' CAM threshold type."
        )

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

        match stage:
            case "fit" | "validate":
                metadata = self._load_df(metadata_uri, cols=TRAIN_METADATA_COLS)

                # --- split train/val ---
                train, val = train_val_split(metadata)
                train = min_count_filter(train, self.dataset_conf.crop_size)
                train_ids = set(train["slide_id"])
                val_ids = set(val["slide_id"])

                # --- load supervision ---
                slide_labels = self._get_slide_labels(metadata)

                cam_uri_train = sup_conf.cam[self.cam_thr_type]
                cam_labels_train = self._load_df(cam_uri_train).pipe(
                    get_subset, train_ids
                )
                annot_labels_train = annot_labels.pipe(get_subset, train_ids)
                sup_train = build_supervision(
                    annot_labels_train, cam_labels_train, slide_labels
                )

                cam_uri_val = sup_conf.cam.annot_restricted_thr
                cam_labels_val = self._load_df(cam_uri_val).pipe(get_subset, val_ids)
                annot_labels_val = annot_labels.pipe(get_subset, val_ids)
                sup_val = build_supervision(
                    annot_labels_val, cam_labels_val, slide_labels
                )

                # --- compute statistics for sampler and normalization ---
                self.positivity = compute_slides_positivity(
                    train_ids,
                    self.supervision_mode,
                    annot_labels_train,
                    cam_labels_train,
                )
                scale_mean = self.dataset_conf.get("scale_mean") or compute_scale_mean(
                    train, efds_path
                )

                self.train = instantiate(
                    self.dataset_conf,
                    metadata=train,
                    scale_mean=scale_mean,
                    supervision=sup_train,
                    supervision_mode=self.supervision_mode,
                    efds_path=efds_path,
                )
                self.val = instantiate(
                    self.dataset_conf,
                    metadata=val,
                    scale_mean=scale_mean,
                    supervision=sup_val,
                    supervision_mode="agreement-strict",
                    efds_path=efds_path,
                    full_slide=True,
                )

            case "test":
                metadata = self._load_df(metadata_uri, cols=BASE_METADATA_COLS)
                slide_labels = self._get_slide_labels(metadata)
                cam_labels = self._load_df(sup_conf.cam.annot_restricted_thr)
                sup = build_supervision(annot_labels, cam_labels, slide_labels)

                self.test = instantiate(
                    self.dataset_conf,
                    metadata=metadata,
                    supervision=sup,
                    scale_mean=self.dataset_conf.scale_mean,
                    efds_path=efds_path,
                    supervision_mode="agreement-strict",
                    full_slide=True,
                )

            case "predict":
                metadata = self._load_df(metadata_uri, cols=BASE_METADATA_COLS)
                slide_labels = self._get_slide_labels(metadata)
                cam_labels = self._load_df(sup_conf.cam.annot_restricted_thr)
                sup = build_supervision(annot_labels, cam_labels, slide_labels)

                self.predict = instantiate(
                    self.dataset_conf,
                    metadata=metadata,
                    supervision=sup,
                    scale_mean=self.dataset_conf.scale_mean,
                    supervision_mode="agreement-strict",
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
