"""Pseudo-positive nuclei for slide-level-supervised training, from crop predictions.

The threshold is a quantile of the nucleus logits over all nuclei of the negative
slides of `threshold_slides`: the validation split (default, 99.9th percentile) or the
training split (99.99th percentile); nuclei of the positive training slides above it
become pseudo-positives. Only slide labels are used.

Saves a parquet with the pseudo-positive nuclei ("slide_id", "id") to `output_path`,
read by the graph classification dataset (`pseudo_labels`), and logs it, the threshold
and per-slide counts to MLflow.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from tqdm import tqdm

from nuclei_graph.data.datamodules.base import split_train_val
from nuclei_graph.mlflow_utils import tag_parent_run


def read_logits(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "nuclei_logit" not in df:  # older predictions only hold probabilities
        p = df["nuclei_prediction"].to_numpy(np.float64)
        df["nuclei_logit"] = np.log(p) - np.log1p(-p)
    return df[["id", "nuclei_logit"]]


DEFAULT_QUANTILES = {"validation": 0.999, "train": 0.9999}


def compute_pseudo_labels(
    train_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    predictions_dir: Path,
    quantile: float,
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """Returns the threshold, the pseudo-positive nuclei and per-slide counts.

    The threshold comes from the negative slides of `threshold_df`, the
    pseudo-positives from the positive slides of `train_df`.
    """
    neg_logits = [
        read_logits(predictions_dir / f"{slide_id}.parquet")["nuclei_logit"].to_numpy()
        for slide_id in tqdm(
            threshold_df.loc[~threshold_df["is_carcinoma"], "slide_id"]
        )
    ]
    threshold = float(np.quantile(np.concatenate(neg_logits), quantile))

    pseudo_dfs, counts = [], []
    for slide_id in tqdm(train_df.loc[train_df["is_carcinoma"], "slide_id"]):
        df = read_logits(predictions_dir / f"{slide_id}.parquet")
        pos = df.loc[df["nuclei_logit"] > threshold, ["id"]].assign(slide_id=slide_id)
        pseudo_dfs.append(pos)
        counts.append(
            {"slide_id": slide_id, "n_nuclei": len(df), "n_pseudo_pos": len(pos)}
        )
    pseudo_df = pd.concat(pseudo_dfs, ignore_index=True)[["slide_id", "id"]]
    return threshold, pseudo_df, pd.DataFrame(counts)


@with_cli_args(["+postprocessing=crop/pseudo_labels"])
@hydra.main(
    config_path="../../configs", config_name="postprocessing", version_base=None
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    assert logger.run_id is not None
    tag_parent_run(logger.run_id, config.get("mlflow_parent_run_id"))

    slides_df = pd.read_parquet(download_artifacts(config.metadata_uri))
    train_df, validation_df = split_train_val(
        slides_df, config.split_size, config.split_stratify_col, config.split_group_col
    )
    assert config.threshold_slides in DEFAULT_QUANTILES, config.threshold_slides
    threshold_df = (
        validation_df if config.threshold_slides == "validation" else train_df
    )
    quantile = config.quantile or DEFAULT_QUANTILES[config.threshold_slides]

    predictions_dir = Path(download_artifacts(config.predictions_uri))
    threshold, pseudo_df, counts_df = compute_pseudo_labels(
        train_df, threshold_df, predictions_dir, quantile
    )

    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pseudo_df.to_parquet(output_path, index=False)

    logger.log_metrics(
        {
            "pseudo_labels/threshold_logit": threshold,
            "pseudo_labels/quantile": quantile,
            "pseudo_labels/n_threshold_neg_slides": int(
                (~threshold_df["is_carcinoma"]).sum()
            ),
            "pseudo_labels/n_pseudo_pos": len(pseudo_df),
            "pseudo_labels/slides_with_pseudo_pos": int(
                (counts_df["n_pseudo_pos"] > 0).sum()
            ),
            "pseudo_labels/n_pos_train_slides": len(counts_df),
        }
    )
    with TemporaryDirectory() as tmp_dir:
        counts_path = Path(tmp_dir) / "pseudo_label_counts.csv"
        counts_df.to_csv(counts_path, index=False)
        logger.experiment.log_artifact(logger.run_id, str(counts_path))
        logger.experiment.log_artifact(logger.run_id, str(output_path))


if __name__ == "__main__":
    main()
