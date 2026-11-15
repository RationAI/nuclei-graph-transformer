"""Slides of iCAIRD cervix to exclude for blur, by the rule used for PANDA - Radboud.

Rule. The Radboud exclusion list (/mnt/projects/prostate_cancer/cancer/qc_masks/PANDA -
Radboud/to_exclude_based_on_qc.csv on the public cluster, used by configs/data/sources/panda.yaml)
was made by notebooks/preprocessing/slide_level_qc_aggregation.ipynb of RationAI/prostate-cancer:
a slide is excluded when `median_coverage(Piqe)` >= 0.2, i.e. the median blurred fraction of its
tissue tiles is at least 20%. That notebook also excludes slides with median residual-artifact
or folding coverage >= 0.3 and slides missing from the QC; only the blur part is applied here.
Whole slides are excluded; nothing is removed at the object level.

Coverage. From qc_metrics.csv of the QC run "QC Masks: iCAIRD Cervix" (public MLflow, run
149a9da90b6046db8ac880b9c4cfc318). The 36 slides of the QC retry have masks but no row in
qc_metrics.csv (the retry did not merge them); for those it is recomputed from blur_per_tile:
the tile values (blurred fraction, 0-255) of the 512 px tiles (at level 1) that hold at least
MIN_OBJECTS objects of the cleaned Cellpose batch. On the 2,488 slides with metrics this
agrees with `median_coverage(Piqe)` on the >= 0.2 decision for 2,481 slides (mean absolute
difference 0.007) and never flags a slide the QC does not.

The 60 slides re-annotated by a pathologist (annotation report, public MLflow run
0fc2b02cf7864c508d2868b25ce7b5e6, selected_slides.csv) must not be excluded: the script
stops if any of them meets the rule.

Writes OUT/to_exclude_based_on_qc.csv (slide_id, slide_path and the measurements, with the
same `slide_path` column as the metadata so it can be passed as `exclude_slides_uris`),
OUT/blur_by_slide.csv (every slide), OUT/blur_exclusion.json (rule and counts), and logs the
three files to a run on the tracking server in the environment (the public MLflow, as for the
other iCAIRD artifacts).

Usage:
    uv run python -m scripts.preprocessing.quality_control.blur_exclusion_icaird_cervix [--no-log]
"""

import argparse
import json
import time
from multiprocessing import Pool

import mlflow
import numpy as np
import pandas as pd
import tifffile
from mlflow.artifacts import download_artifacts

from scripts.preprocessing.nuclei_segmentation.align_stain_features_icaird_cervix_cellpose import (
    OUT as ALIGNED,
)
from scripts.preprocessing.nuclei_segmentation.review_overlays_icaird_cervix_cellpose import (
    OUT as REVIEW,
)
from scripts.preprocessing.nuclei_segmentation.review_overlays_icaird_cervix_cellpose import (
    SEG,
)


QC = REVIEW / "qc_blur"  # masks, qc_metrics.csv and qc_errors.log of the QC run
OUT = QC / "exclusion"
QC_RUN = "149a9da90b6046db8ac880b9c4cfc318"
REVIEW_RUN = "0fc2b02cf7864c508d2868b25ce7b5e6"
METADATA = "mlflow-artifacts:/37/93f193ea5d9a4e3fb5e94c8b5df4bbd0/artifacts/icaird_cervix/slides_metadata.csv"
MAX_MEDIAN_BLUR = 0.2
MASK_DS, TILE_PX, MIN_OBJECTS = (
    8,
    1024,
    20,
)  # mask level 3; 512 px at level 1 = 1024 px at level 0


def tile_coverage(slide_id: str) -> dict:
    """Mean and median blur coverage of the tissue tiles, from blur_per_tile."""
    tiles = tifffile.imread(QC / "blur_per_tile" / f"{slide_id}.tiff")
    tiles = tiles[..., 0] if tiles.ndim == 3 else tiles
    centroids = np.load(ALIGNED / f"{slide_id}.npz")["centroid"]
    step = TILE_PX // MASK_DS
    grid = (-(-tiles.shape[0] // step), -(-tiles.shape[1] // step))
    ij = np.clip(
        np.floor(centroids / TILE_PX).astype(int), 0, [grid[1] - 1, grid[0] - 1]
    )
    counts = np.zeros(grid, int)
    np.add.at(counts, (ij[:, 1], ij[:, 0]), 1)
    yy, xx = np.meshgrid(
        np.minimum(np.arange(grid[0]) * step + step // 2, tiles.shape[0] - 1),
        np.minimum(np.arange(grid[1]) * step + step // 2, tiles.shape[1] - 1),
        indexing="ij",
    )
    values = tiles[yy, xx][counts >= MIN_OBJECTS] / 255
    return {
        "slide_id": slide_id,
        "mean_coverage_piqe": float(values.mean()) if len(values) else np.nan,
        "median_coverage_piqe": float(np.median(values)) if len(values) else np.nan,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--no-log", action="store_true", help="do not log to MLflow")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    slides = pd.read_csv(download_artifacts(METADATA))
    reannotated = set(
        pd.read_csv(
            download_artifacts(f"runs:/{REVIEW_RUN}/selected_slides.csv")
        ).slide_id
    )
    metrics = pd.read_csv(QC / "qc_metrics.csv").rename(
        columns={"slide_name": "slide_id"}
    )
    metrics = metrics.rename(
        columns={
            "mean_coverage(Piqe)": "mean_coverage_piqe",
            "median_coverage(Piqe)": "median_coverage_piqe",
        }
    )[["slide_id", "mean_coverage_piqe", "median_coverage_piqe"]]
    metrics["coverage_from"] = "qc_metrics"
    masks = {p.stem for p in (QC / "blur_per_tile").glob("*.tiff")}
    missing = sorted(masks - set(metrics.slide_id))
    with Pool(8) as pool:
        recomputed = pd.DataFrame(pool.map(tile_coverage, missing))
    recomputed["coverage_from"] = "blur_per_tile"

    table = slides.merge(pd.concat([metrics, recomputed]), on="slide_id", how="left")
    table["coverage_from"] = table.coverage_from.fillna("no_qc")
    table["reannotated_test_slide"] = table.slide_id.isin(reannotated)
    table["has_nuclei"] = table.slide_id.map(lambda s: (SEG / f"slide_id={s}").exists())
    table["excluded"] = table.median_coverage_piqe >= MAX_MEDIAN_BLUR
    table = table.sort_values(["excluded", "median_coverage_piqe"], ascending=False)
    table.to_csv(OUT / "blur_by_slide.csv", index=False)

    excluded = table[table.excluded]
    if excluded.reannotated_test_slide.any():
        raise SystemExit(
            f"re-annotated slides meet the rule: {excluded[excluded.reannotated_test_slide].slide_id.tolist()}"
        )
    excluded.drop(columns=["excluded"]).to_csv(
        OUT / "to_exclude_based_on_qc.csv", index=False
    )
    no_qc = table[table.coverage_from == "no_qc"]
    summary = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "rule": f"median_coverage(Piqe) >= {MAX_MEDIAN_BLUR}: the median blurred fraction of the tissue tiles",
        "rule_source": (
            "RationAI/prostate-cancer notebooks/preprocessing/slide_level_qc_aggregation.ipynb, which made "
            "PANDA - Radboud/to_exclude_based_on_qc.csv; its residual (>= 0.3), folding (>= 0.3) and "
            "missing-QC exclusions are not applied"
        ),
        "level": "slides only; no objects removed",
        "qc_run": QC_RUN,
        "slides": len(table),
        "coverage_from": table.coverage_from.value_counts().to_dict(),
        "excluded": len(excluded),
        "excluded_slides": excluded.slide_id.tolist(),
        "excluded_by_split": excluded.split.value_counts().to_dict(),
        "excluded_by_category": excluded.category.value_counts().to_dict(),
        "reannotated_test_slides_excluded": int(excluded.reannotated_test_slide.sum()),
        "mean_coverage_over_0.2": int(
            (table.mean_coverage_piqe > MAX_MEDIAN_BLUR).sum()
        ),
        "slides_without_qc_not_excluded": no_qc.slide_id.tolist(),
        "slides_without_qc_reannotated": no_qc[
            no_qc.reannotated_test_slide
        ].slide_id.tolist(),
        "script": "scripts/preprocessing/quality_control/blur_exclusion_icaird_cervix.py",
    }
    (OUT / "blur_exclusion.json").write_text(json.dumps(summary, indent=1))
    print(
        excluded[
            [
                "slide_id",
                "split",
                "category",
                "subcategory",
                "mean_coverage_piqe",
                "median_coverage_piqe",
                "coverage_from",
                "reannotated_test_slide",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    print(json.dumps(summary, indent=1))
    if args.no_log:
        return
    mlflow.set_experiment("Nuclei Graph — Transformer")
    with mlflow.start_run(
        run_name="🚫 QC Slide Exclusion (blur): iCAIRD Cervix"
    ) as run:
        mlflow.set_tag(
            "mlflow.note.content", summary["rule"] + "; " + summary["rule_source"]
        )
        mlflow.log_params(
            {
                "max_median_blur": MAX_MEDIAN_BLUR,
                "qc_run": QC_RUN,
                "min_objects_per_tile": MIN_OBJECTS,
            }
        )
        for name in (
            "to_exclude_based_on_qc.csv",
            "blur_by_slide.csv",
            "blur_exclusion.json",
        ):
            mlflow.log_artifact(str(OUT / name), artifact_path="qc_exclusion")
        print(f"logged to run {run.info.run_id}")


if __name__ == "__main__":
    main()
