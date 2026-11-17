from abc import abstractmethod
from collections.abc import Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from hydra.utils import instantiate
from lightning import LightningDataModule
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from ratiopath.model_selection import train_test_split

from nuclei_graph.data.memory import check_system_allocator, release_memory
from nuclei_graph.data.supervision import (
    DatasetSupervision,
    SupervisionStrategy,
    build_supervision,
)


# A metadata URI, or {provider: URI} for slides of several providers (multi-provider
# tables get a `provider` column).
type MetadataUri = str | Mapping[str, str]
# One supervision strategy, or {provider: strategy} for multi-provider tables.
type Strategy = SupervisionStrategy | dict[str, SupervisionStrategy]

PROVIDER_COL = "provider"


def is_provider_map(cfg: object) -> bool:
    """Whether a metadata or strategy config is a {provider: ...} mapping.

    A strategy config is a mapping too, but it has a `_target_`.
    """
    return isinstance(cfg, Mapping | DictConfig) and "_target_" not in cfg


def instantiate_strategy(cfg: DictConfig | None) -> Strategy | None:
    if cfg is None:
        return None
    if is_provider_map(cfg):
        return {str(provider): instantiate(c) for provider, c in cfg.items()}
    return instantiate(cfg)


def strategy_modes(strategy: Strategy) -> list[str]:
    if isinstance(strategy, dict):
        return [s.mode for s in strategy.values()]
    return [strategy.mode]


def split_train_val(
    slides_df: pd.DataFrame,
    split_size: float,
    stratify_col: str | None,
    group_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The fixed (random_state=42) train/validation split of the training slides."""
    train_df, validation_df = train_test_split(
        slides_df,
        test_size=split_size,
        random_state=42,
        stratify=slides_df[stratify_col] if stratify_col else None,
        groups=slides_df[group_col] if group_col else None,
    )
    return train_df.reset_index(drop=True), validation_df.reset_index(drop=True)


def read_supervision(path: str, slide_ids: set[str]) -> pd.DataFrame:
    """The supervision rows of `slide_ids` from the parquet file or directory `path`.

    Only the rows of these slides are read (the per-slide files of the other slides are
    skipped), and string columns stay Arrow-backed, which takes ~3x less memory than
    Python strings (the nuclei ids are 64-character hashes). They are read as
    `large_string`: selecting rows that are not contiguous concatenates a column's
    chunks (`build_supervision.take_rows`), and the 32-bit offsets of `string`
    overflow beyond 2 GB of text (~33M ids). The read is single-threaded so that
    `release_memory` (which frees the calling thread's memory) returns it to the OS.
    """
    dataset = ds.dataset(path, format="parquet", partitioning="hive")
    schema = pa.schema(
        [
            f.with_type(pa.large_string()) if pa.types.is_string(f.type) else f
            for f in dataset.schema
        ],
        metadata=dataset.schema.metadata,
    )
    dataset = ds.dataset(path, schema=schema, format="parquet", partitioning="hive")
    value_set = pa.array(sorted(slide_ids), type=schema.field("slide_id").type)
    table = dataset.to_table(
        filter=pc.field("slide_id").isin(value_set), use_threads=False
    )
    return table.to_pandas(
        types_mapper=lambda t: pd.ArrowDtype(t) if pa.types.is_large_string(t) else None
    )


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
        context: str = "knn",
    ) -> None:
        super().__init__()
        check_system_allocator()  # see `ReleaseMemory`
        self.context = context
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

        # one strategy, or {provider: strategy} for multi-provider metadata
        self.train_strategy = (
            instantiate_strategy(supervision.train_strategy)
            if supervision is not None
            else None
        )
        self.eval_strategy = (
            instantiate_strategy(supervision.eval_strategy)
            if supervision is not None
            else None
        )

    def filter_df(
        self, df: pd.DataFrame | None, slide_ids: set[str]
    ) -> pd.DataFrame | None:
        if df is None:
            return None
        return df[df["slide_id"].isin(slide_ids)].reset_index(drop=True)

    def load_df(self, uri: MetadataUri, cols: list[str] | None = None) -> pd.DataFrame:
        """The slides table of `uri`.

        With {provider: URI}, the providers' tables are concatenated with a `provider`
        column (a slide may belong to one provider only).
        """
        if not is_provider_map(uri):
            assert isinstance(uri, str), uri
            return pd.read_parquet(download_artifacts(uri), columns=cols)
        assert isinstance(uri, Mapping)
        df = pd.concat(
            [
                self.load_df(u, cols).assign(**{PROVIDER_COL: str(provider)})
                for provider, u in uri.items()
            ],
            ignore_index=True,
        )
        assert df["slide_id"].is_unique, "a slide is listed more than once"
        return df

    def get_carcinoma_map(self, slide_df: pd.DataFrame) -> dict[str, bool]:
        return {
            str(k): v for k, v in slide_df.set_index("slide_id")["is_carcinoma"].items()
        }

    def supervise(
        self, slides_df: pd.DataFrame, strategy: Strategy
    ) -> DatasetSupervision:
        """`prepare_supervision` with the slides' strategy (per provider for a mapping)."""
        if not isinstance(strategy, dict):
            return self.prepare_supervision(slides_df, strategy.paths, strategy)
        providers = set(slides_df[PROVIDER_COL])
        assert providers <= set(strategy), (
            f"no supervision strategy for {providers - set(strategy)}"
        )
        supervision_map = {}
        for provider in sorted(providers):
            s = strategy[provider]
            part = slides_df[slides_df[PROVIDER_COL] == provider]
            supervision_map.update(
                self.prepare_supervision(part, s.paths, s).supervision_map
            )
        return DatasetSupervision(supervision_map)

    def prepare_supervision(
        self,
        slides_df: pd.DataFrame,
        sup_paths: dict[str, str | None],
        strategy: SupervisionStrategy,
    ) -> DatasetSupervision:
        ids = set(slides_df["slide_id"])
        sup_dfs = {
            sup: read_supervision(path, ids) if path is not None else None
            for sup, path in sup_paths.items()
        }
        supervision = build_supervision(
            strategy=strategy,
            carcinoma_map=self.get_carcinoma_map(slides_df),
            sup_dfs=sup_dfs,
        )
        # the allocators keep the freed tables (iCAIRD training: ~17 GB vs ~2 GB)
        del sup_dfs
        release_memory()
        return supervision

    def load_train_val_dfs(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """The training and validation slides.

        With a `valid` metadata table and `split_size: null`, the dataset's own validation
        split is used as is (e.g. iCAIRD); otherwise validation is the fixed split of the
        `train` table (`get_train_val_dfs`). With {provider: URI} as `train`, each
        provider's table is split on its own (so a provider's split does not depend on
        the others) and the splits are concatenated with a `provider` column.
        """
        train_uri = self.metadata_uris_cfg["train"]
        if is_provider_map(train_uri):
            assert self.metadata_uris_cfg.get("valid") is None
            splits = [
                tuple(
                    df.assign(**{PROVIDER_COL: str(provider)})
                    for df in self.get_train_val_dfs(self.load_df(uri))
                )
                for provider, uri in train_uri.items()
            ]
            return (
                pd.concat([t for t, _ in splits], ignore_index=True),
                pd.concat([v for _, v in splits], ignore_index=True),
            )
        valid_uri = self.metadata_uris_cfg.get("valid")
        if valid_uri is not None and self.split_size is None:
            train_df = self.load_df(self.metadata_uris_cfg["train"])
            validation_df = self.load_df(valid_uri)
            overlap = set(train_df["slide_id"]) & set(validation_df["slide_id"])
            assert not overlap, f"{len(overlap)} slides in both train and valid"
            return train_df, validation_df
        assert self.split_size is not None, (
            "split_size is required unless a `valid` metadata table is given"
        )
        return self.get_train_val_dfs(self.load_df(self.metadata_uris_cfg["train"]))

    def get_train_val_dfs(
        self, slides_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        assert self.split_size is not None
        return split_train_val(
            slides_df, self.split_size, self.split_stratify_col, self.split_group_col
        )

    @abstractmethod
    def setup(self, stage: str) -> None:
        raise NotImplementedError
