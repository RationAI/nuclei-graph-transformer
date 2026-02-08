from collections.abc import Generator

import pandas as pd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from pandas import DataFrame


def collect_artifact_uris(dataset: DictConfig) -> set[str]:
    """Recursively collects all non-None unique artifact URIs from a dataset configuration.

    Args:
        dataset: A dataset configuration with a `.uris` attribute.

    Returns:
        set[str]: A set of all unique artifact URIs (strings).
    """

    def flatten(conf: DictConfig) -> Generator[str]:
        for v in conf.values():
            if isinstance(v, DictConfig):
                yield from flatten(v)
            elif v is not None:
                yield str(v)

    if dataset.get("uris") is None:
        return set()

    return {uri for uri in flatten(dataset.uris)}


def load_df(uri: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = download_artifacts(uri)
    return pd.read_parquet(path, columns=columns)


def slide_labels_from_df(df: DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in df.set_index("slide_id")["is_carcinoma"].items()}
