"""Nucleus index of the validation and test slides for the concept (SAE) analysis.

Writes, per split, `activations/<split>/index.parquet` with every nucleus
(slide_id, nucleus_id, label, centroid_x_um, centroid_y_um), slides contiguous and
nuclei in `get_nuclei` (id) order; `ActivationsCallback` fills the activation arrays
in the same row order. `slides.parquet` holds the slide metadata for prediction and
each slide's `offset`/`n_nuclei` in the arrays of its split.

The validation slides are those of the training run (same split helper and seed) and
the labels are the datamodule's dense evaluation supervision.

`concept_sample.parquet` (for the probe analysis only) samples at most
`probe_sample.max_per_slide` nuclei per slide, balanced by label (up to half per
label, filled from the other label when one is short), with `sampling_weight` =
(#nuclei of the label in the slide) / (#sampled nuclei of the label in the slide) and
`row` = the nucleus' row in the arrays of its split.
"""

import zlib
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from joblib import Parallel, delayed
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from nuclei_graph.data.datamodules.base import METADATA_COLS_EVAL
from nuclei_graph.data.datasets.crop.base import BaseCropDataset
from nuclei_graph.data.supervision import NucleiSupervision
from nuclei_graph.mlflow_utils import tag_parent_run
from nuclei_graph.xai.runtime import setup_precision


if TYPE_CHECKING:
    from nuclei_graph.data.datamodules.crop import CropDataModule


def ids_checksum(ids: np.ndarray) -> int:
    """Checksum of a slide's nucleus ids in index order."""
    return zlib.crc32("\n".join(map(str, ids)).encode())


def balanced_counts(n_pos: int, n_neg: int, max_total: int) -> tuple[int, int]:
    """Up to max_total / 2 per label; the remainder is filled from the other label."""
    take_pos = min(n_pos, max_total // 2)
    take_neg = min(n_neg, max_total - max_total // 2)
    remaining = max_total - take_pos - take_neg
    extra_pos = min(remaining, n_pos - take_pos)
    take_pos += extra_pos
    take_neg += min(remaining - extra_pos, n_neg - take_neg)
    return take_pos, take_neg


def probe_sample(
    labels: np.ndarray, sup_mask: np.ndarray, max_per_slide: int, seed: int, key: str
) -> pd.DataFrame:
    """Balanced-fill sample of a slide's nuclei: local positions and weights."""
    rng = np.random.default_rng([seed, zlib.crc32(key.encode())])  # order-independent
    pos = np.flatnonzero(sup_mask & (labels == 1))
    neg = np.flatnonzero(sup_mask & (labels == 0))
    take_pos, take_neg = balanced_counts(len(pos), len(neg), max_per_slide)

    chosen, weights = [], []
    for candidates, take in ((pos, take_pos), (neg, take_neg)):
        if take > 0:
            chosen.append(np.sort(rng.choice(candidates, size=take, replace=False)))
            weights.append(np.full(take, len(candidates) / take))
    return pd.DataFrame(
        {"position": np.concatenate(chosen), "sampling_weight": np.concatenate(weights)}
    )


def index_slide(
    slide: pd.Series,
    loader: BaseCropDataset,
    nuclei_sup: NucleiSupervision,
    max_per_slide: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    nuclei = loader.get_nuclei(slide.slide_nuclei_path)
    centroids = loader.get_centroids(nuclei, slide.mpp_x, slide.mpp_y)
    labels = nuclei_sup.get_targets(len(nuclei)).numpy().astype(np.int8)
    sup_mask = nuclei_sup.get_sup_mask(len(nuclei)).numpy()
    assert len(labels) == len(nuclei), f"Labels do not match nuclei: {slide.slide_id}"
    assert sup_mask.all(), f"Labels are not dense: {slide.slide_id}"

    ids = nuclei["id"].to_numpy()
    index = pd.DataFrame(
        {
            "slide_id": slide.slide_id,
            "nucleus_id": ids,
            "label": labels,
            "centroid_x_um": centroids[:, 0].astype(np.float32),
            "centroid_y_um": centroids[:, 1].astype(np.float32),
        }
    )
    sample = probe_sample(labels, sup_mask, max_per_slide, seed, slide.slide_id)
    return index, sample, ids_checksum(ids)


def write_outputs(
    slides: pd.DataFrame,
    results: list[tuple[pd.DataFrame, pd.DataFrame, int]],
    activations_dir: Path,
    out_dir: Path,
) -> dict[str, int]:
    """Writes the per-split index, `slides.parquet` and `concept_sample.parquet`."""
    slides = slides.reset_index(drop=True)
    slides["n_nuclei"] = [len(index) for index, _, _ in results]
    slides["ids_crc32"] = [checksum for _, _, checksum in results]
    slides["offset"] = slides.groupby("split")["n_nuclei"].transform(
        lambda n: n.cumsum() - n
    )

    metrics, samples = {}, []
    for split, split_slides in slides.groupby("split", sort=False):
        rows = split_slides.index.to_numpy()
        index = pd.concat([results[i][0] for i in rows], ignore_index=True)
        index["slide_id"] = index["slide_id"].astype("category")
        split_dir = activations_dir / str(split)
        split_dir.mkdir(parents=True, exist_ok=True)
        index.to_parquet(split_dir / "index.parquet", index=False)

        for i in rows:
            sample = results[i][1]
            row = sample["position"].to_numpy() + slides.at[i, "offset"]
            samples.append(
                index.iloc[row].assign(
                    split=split,
                    sampling_weight=sample["sampling_weight"].to_numpy(),
                    row=row,
                )
            )

        counts = index["label"].value_counts()
        metrics |= {
            f"{split}/slides": len(split_slides),
            f"{split}/nuclei": len(index),
            f"{split}/carcinoma_nuclei": int(counts.get(1, 0)),
            f"{split}/benign_nuclei": int(counts.get(0, 0)),
        }
        print(
            f"{split}: {len(split_slides)} slides, {len(index):,} nuclei, {counts.to_dict()}"
        )
        del index

    out_dir.mkdir(parents=True, exist_ok=True)
    sample = pd.concat(samples, ignore_index=True)
    sample["slide_id"] = sample["slide_id"].astype(str)
    sample.to_parquet(out_dir / "concept_sample.parquet", index=False)
    columns = [*METADATA_COLS_EVAL, "is_carcinoma", "gleason_score", "split"]
    slides[[*columns, "n_nuclei", "offset", "ids_crc32"]].to_parquet(
        out_dir / "slides.parquet", index=False
    )
    return metrics


@with_cli_args(["+xai=prepare_nuclei"])
@hydra.main(config_path="../configs", config_name="xai", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)
    for key, value in setup_precision(config.matmul_precision).items():
        logger.experiment.set_tag(logger.run_id, key, value)

    datamodule: CropDataModule = instantiate(config.data, _recursive_=False)
    assert datamodule.eval_strategy is not None

    # validation split exactly as in `CropDataModule.setup("fit")`
    train_df = datamodule.load_df(datamodule.metadata_uris_cfg["train"])
    _, validation_df = datamodule.get_train_val_dfs(train_df)
    test_df = datamodule.load_df(datamodule.metadata_uris_cfg["test"])
    split_dfs = {"validation": validation_df, "test": test_df}

    slides = pd.concat(
        [split_dfs[split].assign(split=split) for split in config.splits],
        ignore_index=True,
    )
    supervision = datamodule.prepare_supervision(
        slides, datamodule.eval_strategy.paths, datamodule.eval_strategy
    )
    loader = instantiate(config.data.dataset, metadata=slides)

    results = Parallel(n_jobs=-1)(
        delayed(index_slide)(
            slide,
            loader,
            supervision.supervision_map[slide.slide_id].nuclei_supervision,
            config.probe_sample.max_per_slide,
            config.seed,
        )
        for slide in slides.itertuples(index=False)
    )
    for key, value in write_outputs(
        slides, results, Path(config.activations_dir), Path(config.out_dir)
    ).items():
        logger.log_metrics({key: value})
    logger.log_artifact(str(Path(config.out_dir) / "slides.parquet"), "nuclei")
    logger.log_artifact(str(Path(config.out_dir) / "concept_sample.parquet"), "nuclei")


if __name__ == "__main__":
    main()
