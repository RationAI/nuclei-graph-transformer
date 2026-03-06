from collections.abc import Iterable

import pandas as pd
from hydra.utils import instantiate
from lightning import LightningDataModule
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from pandas import DataFrame
from ratiopath.model_selection import train_test_split
from torch.utils.data import DataLoader

from nuclei_graph.data.supervision import (
    DatasetSupervision,
    SupervisionStrategy,
    build_supervision,
)
from nuclei_graph.data.utils import (
    collate_fn,
    collate_fn_predict,
    min_count_filter,
)
from nuclei_graph.nuclei_graph_typing import (
    Batch,
    PredictBatch,
)


BASE_METADATA_COLS = [
    "slide_id",
    "is_carcinoma",
    "slide_nuclei_path",
    "slide_path",
    "mpp_x",
    "mpp_y",
]

TRAIN_METADATA_COLS = [*BASE_METADATA_COLS, "patient_id", "nuclei_count"]


class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        supervision_strategy: DictConfig | None,
        eval_supervision_strategy: DictConfig,
        train_val_split_size: float = 0.1,
        num_workers: int = 0,
        sampler: DictConfig | None = None,
        **data_params: DictConfig,
    ) -> None:
        """Lightning DataModule for nuclei point cloud datasets with weak supervision.

        Args:
            batch_size: Batch size for training.
            num_workers: Number of workers for data loading. Defaults to 0.
            supervision_strategy: A DictConfig defining the type of supervision to use for positive slides during training.
            eval_supervision_strategy: A DictConfig defining the type of supervision to use for evaluation.
            train_val_split_size: Proportion of the training data to use for validation. Defaults to 0.1.
            sampler: Sampler configuration for training data loader. Defaults to None.
            **data_params: Additional parameters expected to contain keys:
                - dataset: DictConfig for instantiation of a Torch Dataset.
                - mlflow_uris: DictConfig with MLflow keys "supervision" and "metadata" containing URIs for respective artifacts.
        """
        super().__init__()
        self.batch_size = batch_size
        self.sup_strategy = instantiate(supervision_strategy)
        self.eval_sup_strategy = instantiate(eval_supervision_strategy)
        self.train_val_split_size = train_val_split_size
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

    def _get_subset(self, df: pd.DataFrame, ids: set[str]) -> pd.DataFrame:
        return df[df["slide_id"].isin(ids)]

    def _prepare_supervision(
        self,
        df: DataFrame,
        strategy: SupervisionStrategy,
        cam_uri: str | None,
        annot_uri: str,
    ) -> DatasetSupervision:
        slide_ids = set(df["slide_id"])

        cam_labels = (
            self._load_df(cam_uri).pipe(self._get_subset, slide_ids)
            if cam_uri is not None
            else None
        )
        annot_labels = self._load_df(annot_uri).pipe(self._get_subset, slide_ids)

        return build_supervision(
            sup_strategy=strategy,
            df_annot=annot_labels,
            df_cam=cam_labels,
            label_map=self._get_slide_labels(df),
        )

    def setup(self, stage: str) -> None:
        mode = "train" if stage in {"fit", "validate"} else stage
        assert self.sup_strategy is not None
        metadata_uri = self.uris_cfg.metadata[mode]

        match stage:
            case "fit" | "validate":
                metadata = self._load_df(metadata_uri, cols=TRAIN_METADATA_COLS)
                metadata = metadata.sort_values(by="slide_id").reset_index(drop=True)

                # --- train/val split ---
                train, val = train_test_split(
                    metadata,
                    test_size=self.train_val_split_size,
                    random_state=42,
                    stratify=metadata["is_carcinoma"],
                    groups=metadata["patient_id"],
                )
                train = train.reset_index(drop=True)
                val = val.reset_index(drop=True)

                train = min_count_filter(train, self.dataset_cfg.crop_size)

                # --- load supervision ---
                sup_train = self._prepare_supervision(
                    df=train,
                    strategy=self.sup_strategy,
                    cam_uri=self.sup_strategy.cam_uri,
                    annot_uri=self.sup_strategy.annot_uri,
                )
                sup_val = self._prepare_supervision(
                    df=val,
                    strategy=self.eval_sup_strategy,
                    cam_uri=self.eval_sup_strategy.cam_uri,
                    annot_uri=self.eval_sup_strategy.annot_uri,
                )

                # --- compute statistics for sampler ---
                self.positivity = {
                    slide_id: slide_sup.nuclei_supervision.get_positivity()
                    for slide_id, slide_sup in sup_train.supervision_map.items()
                }

                # --- instantiate datasets ---
                self.train = instantiate(
                    self.dataset_cfg,
                    metadata=train,
                    supervision=sup_train,
                )
                self.val = instantiate(
                    self.dataset_cfg,
                    metadata=val,
                    supervision=sup_val,
                    full_slide=True,
                )

            case "test" | "predict":
                metadata = self._load_df(metadata_uri, cols=BASE_METADATA_COLS)
                sup = self._prepare_supervision(
                    df=metadata,
                    strategy=self.eval_sup_strategy,
                    cam_uri=self.eval_sup_strategy.cam_uri,
                    annot_uri=self.eval_sup_strategy.annot_uri,
                )
                dataset = instantiate(
                    self.dataset_cfg,
                    metadata=metadata,
                    supervision=sup,
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
