from collections.abc import Iterable

import pandas as pd
from hydra.utils import instantiate
from lightning import LightningDataModule
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
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
        eval_strategy: DictConfig,
        train_strategy: DictConfig | None = None,
        split_size: float = 0.1,
        num_workers: int = 0,
        max_eval_workers: int = 2,
        sampler: DictConfig | None = None,
        **data_params: DictConfig,
    ) -> None:
        """Lightning DataModule for nuclei point cloud datasets with weak supervision.

        Args:
            batch_size: Batch size for training.
            eval_strategy: A DictConfig defining the type of supervision to use for evaluation (validation or test/predict).
            train_strategy: A DictConfig defining the type of supervision to use for positive slides during training.
            split_size: Proportion of the training data to use for validation. Defaults to 0.1.
            num_workers: Number of workers for data loading. Defaults to 0.
            max_eval_workers: Maximum number of workers for evaluation data loading. Defaults to 2.
            sampler: Sampler configuration for training data loader. Defaults to None.
            **data_params: Additional parameters expected to contain keys:
                - dataset: DictConfig for instantiation of a Torch Dataset.
                - mlflow_uris: DictConfig with MLflow keys "supervision" and "metadata" containing URIs for respective artifacts.
        """
        super().__init__()
        self.batch_size = batch_size
        self.train_strategy = instantiate(train_strategy)
        self.eval_strategy = instantiate(eval_strategy)
        self.split_size = split_size
        self.num_workers = num_workers
        self.max_eval_workers = max_eval_workers
        self.sampler_cfg = sampler
        self.dataset_cfg = data_params["dataset"]
        self.mlflow_uris_cfg = data_params["mlflow_uris"]
        self.positivity: dict[str, float] = {}

    def _load_df(
        self,
        uri: str | None,
        slide_ids: set[str] | None = None,
        cols: list[str] | None = None,
    ) -> pd.DataFrame | None:
        if uri is None:
            return None
        df = pd.read_parquet(download_artifacts(uri), columns=cols)
        if slide_ids is not None:
            return df[df["slide_id"].isin(slide_ids)].reset_index(drop=True)
        return df

    def _get_supervision(
        self,
        strategy: SupervisionStrategy,
        slide_df: pd.DataFrame,
        slide_ids: set[str] | None = None,
    ) -> DatasetSupervision:
        slide_label_map = {
            str(k): int(v)
            for k, v in slide_df.set_index("slide_id")["is_carcinoma"].items()
        }
        return build_supervision(
            sup_strategy=strategy,
            label_map=slide_label_map,
            df_annot=self._load_df(strategy.annot_uri, slide_ids),
            df_cam=self._load_df(strategy.cam_uri, slide_ids),
            df_dense=self._load_df(strategy.dense_uri, slide_ids),
        )

    def setup(self, stage: str) -> None:
        mode = "train" if stage in {"fit", "validate"} else stage
        slides_uri = self.mlflow_uris_cfg.metadata[mode]

        match stage:
            case "fit" | "validate":
                assert self.train_strategy is not None

                slides_df = self._load_df(slides_uri, cols=TRAIN_METADATA_COLS)
                assert slides_df is not None

                train_df, validation_df = train_test_split(
                    slides_df,
                    test_size=self.split_size,
                    random_state=42,
                    stratify=slides_df["is_carcinoma"],
                    groups=slides_df["patient_id"],
                )

                train_df = train_df.reset_index(drop=True)
                train_df = min_count_filter(train_df, self.dataset_cfg.crop_size)
                train_sup = self._get_supervision(
                    self.train_strategy, train_df, set(train_df["slide_id"])
                )
                self.positivity = train_sup.positivity_map
                self.train_dataset = instantiate(
                    self.dataset_cfg,
                    slides=train_df,
                    supervision=train_sup,
                )

                validation_df = validation_df.reset_index(drop=True)
                validation_sup = self._get_supervision(
                    self.eval_strategy, validation_df, set(validation_df["slide_id"])
                )
                self.validation_dataset = instantiate(
                    self.dataset_cfg,
                    slides=validation_df,
                    supervision=validation_sup,
                    full_slide=True,
                )

            case "test" | "predict":
                slides_df = self._load_df(slides_uri, cols=BASE_METADATA_COLS)
                assert slides_df is not None
                ids = set(slides_df["slide_id"])
                sup = self._get_supervision(self.eval_strategy, slides_df, ids)

                dataset = instantiate(
                    self.dataset_cfg,
                    slides=slides_df,
                    supervision=sup,
                    full_slide=True,
                    predict=(stage == "predict"),
                )
                if stage == "test":
                    self.test_dataset = dataset
                else:
                    self.predict_dataset = dataset

    def train_dataloader(self) -> Iterable[Batch]:
        sampler = None
        if self.sampler_cfg is not None:
            partial = instantiate(self.sampler_cfg, slides_positivity=self.positivity)
            sampler = partial(dataset=self.train_dataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            collate_fn=collate_fn,
            drop_last=True,
            prefetch_factor=2,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable[Batch]:
        return DataLoader(
            self.validation_dataset,
            batch_size=1,
            num_workers=self.max_eval_workers,
            persistent_workers=self.max_eval_workers > 0,
            prefetch_factor=2 if self.max_eval_workers > 0 else None,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> Iterable[Batch]:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.max_eval_workers,
            persistent_workers=self.max_eval_workers > 0,
            prefetch_factor=2 if self.max_eval_workers > 0 else None,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self) -> Iterable[PredictBatch]:
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            num_workers=self.max_eval_workers,
            persistent_workers=self.max_eval_workers > 0,
            prefetch_factor=2 if self.max_eval_workers > 0 else None,
            collate_fn=collate_fn_predict,
        )
