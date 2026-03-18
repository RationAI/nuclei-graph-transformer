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
        split_size: float,
        num_workers: int,
        max_eval_workers: int,
        mlflow_uris: DictConfig,
        dataset: DictConfig,
        supervision: DictConfig,
        sampler: DictConfig | None = None,
    ) -> None:
        """Lightning DataModule for nuclei point cloud datasets with weak supervision.

        Args:
            batch_size: Batch size for training.
            split_size: Proportion of the training data to use for validation.
            num_workers: Number of workers for data loading.
            max_eval_workers: Maximum number of workers for evaluation data loading.
            mlflow_uris: A DictConfig containing the MLflow URIs for metadata and supervision DataFrames.
            dataset: A DictConfig defining the dataset configuration to instantiate.
            supervision: A DictConfig containing the training and evaluation supervision strategies.
            sampler: Sampler configuration for training data loader. Defaults to None.
        """
        super().__init__()
        self.batch_size = batch_size
        self.train_strategy = instantiate(supervision.train_strategy)
        self.eval_strategy = instantiate(supervision.eval_strategy)
        self.split_size = split_size
        self.num_workers = num_workers
        self.max_eval_workers = max_eval_workers
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

    def _get_label_map(self, slide_df: pd.DataFrame) -> dict[str, int]:
        return {
            str(k): int(v)
            for k, v in slide_df.set_index("slide_id")["is_carcinoma"].items()
        }

    def _load_supervision_dfs(
        self, *strategies: SupervisionStrategy
    ) -> dict[str, pd.DataFrame | None]:
        """Loads supervision DataFrames required by the provided strategies avoiding repeated downloads.

        Assumes that for each supervision type, the URI is identical across all provided strategies.

        Args:
            *strategies: List of SupervisionStrategy objects.

        Returns:
            dict[str, pd.DataFrame | None]: A dictionary containing the loaded DataFrames
                with keys "df_annot", "df_cam", and "df_dense". Values are None if no
                corresponding URI was found across any of the provided strategies.
        """

        def _get_shared_uri(
            strategies: list[SupervisionStrategy], attr: str
        ) -> str | None:
            uris = {getattr(s, attr) for s in strategies if getattr(s, attr)}
            assert len(uris) <= 1, f"Multiple URIs found for {attr}."
            return next(iter(uris)) if uris else None

        valid = list(filter(None, strategies))
        return {
            "df_annot": self._load_df(_get_shared_uri(valid, "annot_uri")),
            "df_cam": self._load_df(_get_shared_uri(valid, "cam_uri")),
            "df_dense": self._load_df(_get_shared_uri(valid, "dense_uri")),
        }

    def _prepare_supervision(
        self,
        slides_df: pd.DataFrame,
        sup_dfs: dict[str, pd.DataFrame | None],
        strategy: SupervisionStrategy,
    ) -> DatasetSupervision:
        ids = set(slides_df["slide_id"])
        filtered = {k: self._filter_df(v, ids) for k, v in sup_dfs.items()}

        return build_supervision(
            sup_strategy=strategy,
            label_map=self._get_label_map(slides_df),
            **filtered,
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
                validation_df = validation_df.reset_index(drop=True)

                train_df = min_count_filter(train_df, self.dataset_cfg.crop_size)
                sup_dfs = self._load_supervision_dfs(
                    self.train_strategy, self.eval_strategy
                )
                train_sup = self._prepare_supervision(
                    train_df, sup_dfs, self.train_strategy
                )
                self.positivity = train_sup.positivity_map
                self.train_dataset = instantiate(
                    self.dataset_cfg,
                    slides=train_df,
                    supervision=train_sup,
                )

                validation_sup = self._prepare_supervision(
                    validation_df, sup_dfs, self.eval_strategy
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
                sup_dfs = self._load_supervision_dfs(self.eval_strategy)
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
