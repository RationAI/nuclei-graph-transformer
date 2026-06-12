from abc import abstractmethod

import pandas as pd
from hydra.utils import instantiate
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from ratiopath.model_selection import train_test_split
from torch.utils.data import Dataset
from lightning import LightningDataModule

from nuclei_graph.data.supervision import (
    DatasetSupervision,
    SupervisionStrategy,
    build_supervision,
)
from nuclei_graph.nuclei_graph_typing import Sample


METADATA_COLS_EVAL = [
    "slide_id",
    "slide_nuclei_path",
    "slide_path",
    "mpp_x",
    "mpp_y",
]


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        eval_num_workers: int,
        metadata: DictConfig,
        dataset: DictConfig,
        block_size: int,
        k: int,
        supervision: DictConfig | None = None,
        split_stratify_col: str | None = None,
        split_group_col: str | None = None,
        split_size: float | None = None,
        sampler: DictConfig | None = None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.block_size = block_size
        self.k = k
        self.num_workers = num_workers
        self.eval_num_workers = eval_num_workers
        self.split_stratify_col = split_stratify_col
        self.split_group_col = split_group_col
        self.split_size = split_size

        self.dataset_cfg = dataset
        self.sampler_cfg = sampler
        self.metadata_uris_cfg = metadata

        self.train_strategy = (
            instantiate(supervision.train_strategy)
            if supervision is not None and supervision.train_strategy is not None
            else None
        )
        self.eval_strategy = (
            instantiate(supervision.eval_strategy) if supervision is not None else None
        )

    def filter_df(
        self, df: pd.DataFrame | None, slide_ids: set[str]
    ) -> pd.DataFrame | None:
        if df is None:
            return None
        return df[df["slide_id"].isin(slide_ids)].reset_index(drop=True)

    def load_df(self, uri: str, cols: list[str] | None = None) -> pd.DataFrame:
        return pd.read_parquet(download_artifacts(uri), columns=cols)

    def get_carcinoma_map(self, slide_df: pd.DataFrame) -> dict[str, bool]:
        return {
            str(k): v for k, v in slide_df.set_index("slide_id")["is_carcinoma"].items()
        }

    def prepare_supervision(
        self,
        slides_df: pd.DataFrame,
        sup_paths: dict[str, str | None],
        strategy: SupervisionStrategy,
    ) -> DatasetSupervision:
        ids = set(slides_df["slide_id"])
        sup_dfs = {
            sup: pd.read_parquet(path) if path is not None else None
            for sup, path in sup_paths.items()
        }
        return build_supervision(
            strategy=strategy,
            carcinoma_map=self.get_carcinoma_map(slides_df),
            sup_dfs={k: self.filter_df(v, ids) for k, v in sup_dfs.items()},
        )

    def get_train_val_dfs(
        self, slides_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df, validation_df = train_test_split(
            slides_df,
            test_size=self.split_size,
            random_state=42,
            stratify=slides_df[self.split_stratify_col]
            if self.split_stratify_col
            else None,
            groups=slides_df[self.split_group_col] if self.split_group_col else None,
        )
        return train_df.reset_index(drop=True), validation_df.reset_index(drop=True)

    @abstractmethod
    def setup(self, stage: str) -> None:
        raise NotImplementedError
