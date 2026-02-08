from collections.abc import Generator

import pandas as pd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig


def collect_artifact_uris(uris: DictConfig | None) -> set[str]:
    """Recursively collects all non-None unique artifact URIs from a given configuration."""
    if uris is None:
        return set()

    def flatten(conf: DictConfig) -> Generator[str]:
        for v in conf.values():
            if isinstance(v, DictConfig):
                yield from flatten(v)
            elif v is not None:
                yield str(v)

    return set(flatten(uris))


def load_df(uri: str, cols: list[str] | None = None) -> pd.DataFrame:
    path = download_artifacts(uri)
    return pd.read_parquet(path, columns=cols)
