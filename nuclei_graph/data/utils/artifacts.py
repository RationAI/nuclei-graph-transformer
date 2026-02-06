import pandas as pd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig


def collect_artifact_uris(datasets: dict[str, DictConfig]) -> set[str]:
    """Recursively collects all non-None unique artifact URIs from a nested dataset configuration.

    Args:
        datasets: A dictionary mapping dataset stage names to their configurations.

    Returns:
        set[str]: A set of all unique artifact URIs (as strings) found in all datasets.
    """

    def flatten(conf: DictConfig):
        for v in conf.values():
            if isinstance(v, DictConfig):
                yield from flatten(v)
            elif v is not None:
                yield str(v)

    return {
        uri
        for conf in datasets.values()
        if isinstance(conf, DictConfig) and conf.get("uris") is not None
        for uri in flatten(conf.uris)
        if uri is not None
    }


def load_df(uri: str, columns: list[str] | None = None):
    path = download_artifacts(uri)
    return pd.read_parquet(path, columns=columns)
