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
    min_positive_count_filter,
)
from nuclei_graph.nuclei_graph_typing import (
    Batch,
    PredictBatch,
)


METADATA_COLS_EVAL = [
    "slide_id",
    "is_carcinoma",
    "slide_nuclei_path",
    "slide_path",
    "mpp_x",
    "mpp_y",
]


class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        eval_num_workers: int,
        mlflow_uris: DictConfig,
        dataset: DictConfig,
        supervision: DictConfig,
        split_stratify_col: str | None = None,
        split_group_col: str | None = None,
        split_size: float | None = None,
        sampler: DictConfig | None = None,
    ) -> None:
        """Lightning DataModule for nuclei point cloud datasets with weak supervision.

        Args:
            batch_size: Batch size for training.
            num_workers: Number of workers for data loading.
            eval_num_workers: Maximum number of workers for evaluation data loading.
            mlflow_uris: A DictConfig containing the MLflow URIs for metadata and supervision DataFrames.
            dataset: A DictConfig defining the dataset configuration to instantiate.
            supervision: A DictConfig containing the training and evaluation supervision strategies.
            split_stratify_col: Column name to use for stratified splitting. 
                If None, no stratification is applied. Defaults to None.
            split_group_col: Column name to use for group-wise splitting. 
                If None, no group-wise splitting is applied. Defaults to None.
            split_size: Proportion of the training data to use for validation.
            sampler: Sampler configuration for training data loader. Defaults to None.
        """
        super().__init__()
        self.batch_size = batch_size
        self.train_strategy = (
            instantiate(supervision.train_strategy)
            if supervision.train_strategy is not None
            else None
        )
        self.split_stratify_col = split_stratify_col
        self.split_group_col = split_group_col
        self.eval_strategy = instantiate(supervision.eval_strategy)
        self.split_size = split_size
        self.num_workers = num_workers
        self.eval_num_workers = eval_num_workers
        self.sampler_cfg = sampler
        self.dataset_cfg = dataset
        self.mlflow_uris_cfg = mlflow_uris
        self.positivity: dict[str, float] = {}

    def _filter_df(
        self, df: pd.DataFrame | None, slide_ids: set[str]
    ) -> pd.DataFrame | None:
        if df is None:
            return None
        return df[df["slide_id"].isin(slide_ids)].reset_index(drop=True)

    def _load_df(
        self, uri: str | None, cols: list[str] | None = None
    ) -> pd.DataFrame | None:
        if uri is None:
            return None
        return pd.read_parquet(download_artifacts(uri), columns=cols)

    def _get_carcinoma_map(self, slide_df: pd.DataFrame) -> dict[str, bool]:
        return {
            str(k): v for k, v in slide_df.set_index("slide_id")["is_carcinoma"].items()
        }

    def _load_sup_sources(
        self, strategy: SupervisionStrategy
    ) -> dict[str, pd.DataFrame | None]:
        return {uri_key: self._load_df(uri) for uri_key, uri in strategy.uris.items()}

    def _prepare_supervision(
        self,
        slides_df: pd.DataFrame,
        sup_dfs: dict[str, pd.DataFrame | None],
        strategy: SupervisionStrategy,
    ) -> DatasetSupervision:
        ids = set(slides_df["slide_id"])
        return build_supervision(
            strategy=strategy,
            carcinoma_map=self._get_carcinoma_map(slides_df),
            sup_dfs={k: self._filter_df(v, ids) for k, v in sup_dfs.items()},
        )

    def setup(self, stage: str) -> None:
        mode = "train" if stage in {"fit", "validate"} else stage
        slides_uri = self.mlflow_uris_cfg.metadata[mode]

        match stage:
            case "fit" | "validate":
                assert self.split_size is not None
                slides_df = self._load_df(slides_uri)
                assert slides_df is not None

                train_df, validation_df = train_test_split(
                    slides_df,
                    test_size=self.split_size,
                    random_state=42,
                    stratify=slides_df[self.split_stratify_col]
                    if self.split_stratify_col
                    else None,
                    groups=slides_df[self.split_group_col]
                    if self.split_group_col
                    else None,
                )
                train_df = train_df.reset_index(drop=True)
                validation_df = validation_df.reset_index(drop=True)

                if stage == "fit":
                    assert self.train_strategy is not None

                    train_df = min_count_filter(train_df, self.dataset_cfg.crop_size)
                    train_sup_dfs = self._load_sup_sources(self.train_strategy)
                    train_sup = self._prepare_supervision(
                        train_df, train_sup_dfs, self.train_strategy
                    )
                    self.positivity = train_sup.positivity_map

                    if self.dataset_cfg.mil:
                        min_pos_count = (
                            self.dataset_cfg.crop_size * self.dataset_cfg.crop_pos_thr
                        )
                        train_df = min_positive_count_filter(
                            train_df, min_pos_count, train_sup.pos_count_map
                        )
                    self.train_dataset = instantiate(
                        self.dataset_cfg,
                        slides=train_df,
                        supervision=train_sup,
                    )

                validation_sup_dfs = self._load_sup_sources(self.eval_strategy)
                validation_sup = self._prepare_supervision(
                    validation_df, validation_sup_dfs, self.eval_strategy
                )
                self.validation_dataset = instantiate(
                    self.dataset_cfg,
                    slides=validation_df,
                    supervision=validation_sup,
                    full_slide=True,
                )

            case "test" | "predict":
                slides_df = self._load_df(slides_uri, cols=METADATA_COLS_EVAL)
                assert slides_df is not None
                sup_dfs = self._load_sup_sources(self.eval_strategy)
                sup = self._prepare_supervision(slides_df, sup_dfs, self.eval_strategy)
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
            prefetch_factor=2 if self.num_workers > 0 else None,
            pin_memory=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable[Batch]:
        return DataLoader(
            self.validation_dataset,
            batch_size=1,
            num_workers=self.eval_num_workers,
            persistent_workers=self.eval_num_workers > 0,
            prefetch_factor=2 if self.eval_num_workers > 0 else None,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> Iterable[Batch]:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.eval_num_workers,
            persistent_workers=self.eval_num_workers > 0,
            prefetch_factor=2 if self.eval_num_workers > 0 else None,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self) -> Iterable[PredictBatch]:
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            num_workers=self.eval_num_workers,
            persistent_workers=self.eval_num_workers > 0,
            prefetch_factor=2 if self.eval_num_workers > 0 else None,
            collate_fn=collate_fn_predict,
        )
