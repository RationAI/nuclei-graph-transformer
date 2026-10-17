"""Whole-slide concept activation masks of the blinded review's concepts, for xOpat.

For each concept in `letters` (named `concept_<letter>` as in the deck of
`xai.session_material`) and each test slide, a single-channel uint8 BigTIFF with the
nucleus polygons filled by the concept code c (the SAE activation, not the contribution):
0 if c_ij = 0, else max(1, round(255 · min(c_ij / s_j, 1))), where s_j is the
`scale_quantile` quantile of the positive c_j over all test nuclei. 0 is shared by the
background and inactive nuclei. All masks of a slide are drawn in one pass over its
polygons (level-0 pixels, divided by the level's downsample).

Logged artifacts: `concept_masks/<label>/<slide_id>.tiff`, `slide_summary.csv` (one row
per slide: carcinoma label and share, the shares of nuclei with c_j > 0 and with
c_j > `high_fraction` · s_j, mask geometry, status, peak memory) and `scales.json`
(label -> s_j). The label -> SAE index key is written to `key_path` only: nothing logged
tells a concept's index, sign or logit weight.
"""

import json
import resource
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import hydra
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyvips
import ray
from omegaconf import DictConfig
from PIL import Image, ImageDraw
from rationai.masks import process_items, write_big_tiff
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.openslide import OpenSlide

from nuclei_graph.mlflow_utils import tag_parent_run


SCALING_RULE = "0 if c == 0 else max(1, round(255 * min(c / s_j, 1)))"


def concept_key(deck_key: str, letters: list[str]) -> pd.DataFrame:
    """Labels (`concept_<letter>`) of the deck's letters and their SAE indices."""
    key = pd.read_csv(deck_key).set_index("letter")
    missing = set(letters) - set(key.index)
    assert not missing, f"Letters not in the deck: {sorted(missing)}"
    key = key.loc[letters, ["section", "concept"]].reset_index()
    key.insert(0, "label", "concept_" + key["letter"])
    return key


def pixel_values(c: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """0 for c = 0, else max(1, round(255 · min(c / s, 1)))."""
    values = np.maximum(np.rint(255 * np.minimum(c / scales, 1.0)), 1)
    return np.where(c > 0, values, 0).astype(np.uint8)


def slide_stats(
    slides: pd.DataFrame,
    c: np.ndarray,
    carcinoma: np.ndarray,
    scales: np.ndarray,
    labels: list[str],
    high_fraction: float,
) -> pd.DataFrame:
    starts = slides["offset"].to_numpy()
    n = slides["n_nuclei"].to_numpy()
    stats = slides[["slide_id", "n_nuclei", "is_carcinoma", "gleason_score"]].copy()
    stats["carcinoma_fraction"] = (
        np.add.reduceat(carcinoma.astype(np.float64), starts) / n
    )
    active = np.add.reduceat((c > 0).astype(np.int64), starts) / n[:, None]
    high = np.add.reduceat((c > high_fraction * scales).astype(np.int64), starts)
    for j, label in enumerate(labels):
        stats[f"{label}_active_fraction"] = active[:, j]
        stats[f"{label}_high_fraction"] = high[:, j] / n
    return stats


def render_slide(
    item: dict[str, Any],
    labels: list[str],
    level: int,
    tile_width: int,
    tile_height: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Writes the masks of one slide; `item["values"]` is (nuclei x labels) in index order."""
    nuclei = pd.read_parquet(item["slide_nuclei_path"], columns=["id", "polygon"])
    nuclei = nuclei.sort_values("id").reset_index(drop=True)  # index (get_nuclei) order
    ids = nuclei["id"].to_numpy()
    assert len(np.unique(ids)) == len(ids), f"Duplicate nucleus ids: {item['slide_id']}"
    assert np.array_equal(ids, item["nucleus_ids"]), (
        f"Nuclei do not match the index: {item['slide_id']}"
    )

    with OpenSlide(item["slide_path"]) as slide:
        mask_size = slide.level_dimensions[level]
        mask_mpp_x, mask_mpp_y = slide.slide_resolution(level)
        downsample = slide.level_downsamples[level]

    masks = {label: Image.new("L", size=mask_size) for label in labels}
    canvases = [ImageDraw.Draw(mask) for mask in masks.values()]
    for polygon, values in zip(nuclei["polygon"], item["values"], strict=True):
        if not values.any():
            continue
        xy = (np.asarray(polygon, dtype=np.float64) / downsample).tolist()
        for canvas, value in zip(canvases, values.tolist(), strict=True):
            if value:
                canvas.polygon(xy=xy, outline=value, fill=value)

    for label in labels:
        mask = masks.pop(label)
        path = output_dir / label / Path(item["slide_path"]).with_suffix(".tiff").name
        path.parent.mkdir(parents=True, exist_ok=True)
        write_big_tiff(
            image=pyvips.Image.new_from_array(np.array(mask)),
            path=path,
            mpp_x=mask_mpp_x,
            mpp_y=mask_mpp_y,
            tile_width=tile_width,
            tile_height=tile_height,
        )
    return {
        "level": level,
        "downsample": downsample,
        "mask_width": mask_size[0],
        "mask_height": mask_size[1],
        "mpp_x": mask_mpp_x,
        "mpp_y": mask_mpp_y,
    }


@ray.remote(max_calls=1)  # a fresh worker per slide: ru_maxrss is the slide's peak
def process_slide(item: dict[str, Any], records_dir: Path, **kwargs: Any) -> None:
    start = time.monotonic()
    try:
        record = {"status": "ok", "error": "", **render_slide(item, **kwargs)}
    except Exception as e:  # noqa: BLE001 - recorded as a failed slide
        record = {"status": "failed", "error": repr(e)}
    record["seconds"] = time.monotonic() - start
    record["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    (records_dir / f"{item['slide_id']}.json").write_text(
        json.dumps({"slide_id": item["slide_id"], **record})
    )


def code_commit() -> tuple[str, bool]:
    repo = Path(__file__).parent
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return commit, bool(dirty)


@with_cli_args(["+xai=concept_masks"])
@hydra.main(config_path="../configs", config_name="xai", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)

    key = concept_key(config.deck_key, list(config.letters))
    labels = key["label"].tolist()
    key_path = Path(config.key_path)  # local only, never logged
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key.to_csv(key_path, index=False)

    slides = pd.read_parquet(config.slides)
    slides = slides[slides["split"] == "test"].sort_values("offset")
    index = pq.read_table(
        Path(config.activations_dir) / "test" / "index.parquet",
        columns=["nucleus_id", "label"],
    )
    codes = np.load(Path(config.concept_dir) / "codes_test.npy", mmap_mode="r")
    assert len(index) == len(codes) == slides["n_nuclei"].sum()
    c = np.asarray(codes)[:, key["concept"].to_numpy()]

    scales = np.array(
        [
            np.quantile(c[c[:, j] > 0, j], config.scale_quantile)
            for j in range(c.shape[1])
        ]
    )
    values = pixel_values(c, scales)
    stats = slide_stats(
        slides,
        c,
        index.column("label").to_numpy(),
        scales,
        labels,
        config.high_fraction,
    )
    nucleus_ids = index.column("nucleus_id")

    if config.slide_ids is not None:
        slides = slides[slides["slide_id"].isin(config.slide_ids)]
    items = [
        {
            "slide_id": slide.slide_id,
            "slide_path": slide.slide_path,
            "slide_nuclei_path": slide.slide_nuclei_path,
            "nucleus_ids": nucleus_ids.slice(slide.offset, slide.n_nuclei).to_numpy(),
            "values": values[slide.offset : slide.offset + slide.n_nuclei],
        }
        for slide in slides.itertuples(index=False)
    ]

    with TemporaryDirectory() as tmp:
        output_dir, records_dir = Path(tmp) / "concept_masks", Path(tmp) / "records"
        records_dir.mkdir()
        process_items(
            items=items,
            process_item=process_slide.options(memory=config.task_memory_gb * 1024**3),
            fn_kwargs={
                "records_dir": records_dir,
                "labels": labels,
                "level": config.level,
                "tile_width": config.mask_tile_width,
                "tile_height": config.mask_tile_height,
                "output_dir": output_dir,
            },
            max_concurrent=config.max_concurrent,
        )
        records = pd.DataFrame(
            [json.loads(path.read_text()) for path in records_dir.glob("*.json")]
        )
        summary = stats.merge(records, on="slide_id", how="inner")
        summary.to_csv(Path(tmp) / "slide_summary.csv", index=False)
        (Path(tmp) / "scales.json").write_text(
            json.dumps(dict(zip(labels, scales.tolist(), strict=True)), indent=2)
        )

        failed = summary[summary["status"] != "ok"]
        for row in failed.itertuples(index=False):
            print(f"failed: {row.slide_id}: {row.error}")
        models = pd.read_csv(Path(config.sae_dir) / "models.csv")
        sae = models[
            (models["method"] == "sae")
            & (models["k"] == config.num_concepts)
            & (models["seed"] == config.sae_seed)
        ].iloc[0]
        commit, dirty = code_commit()
        logger.log_hyperparams(
            {
                "sae_lambda": float(sae["l1_coeff"]),
                "sae_model": sae["path"],
                "code_commit": commit,
                "code_dirty": dirty,
                "slides_processed": int((summary["status"] == "ok").sum()),
                "slides_failed": len(failed),
            }
        )
        logger.log_artifacts(str(output_dir), "concept_masks")
        logger.log_artifact(str(Path(tmp) / "slide_summary.csv"))
        logger.log_artifact(str(Path(tmp) / "scales.json"))


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
