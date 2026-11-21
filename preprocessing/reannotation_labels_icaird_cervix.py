"""Test labels of the iCAIRD cervix slides re-annotated by a pathologist.

For each positive test slide with the pathologist's regions (one 
`annotations/<slide_id>.tiff.json` per slide in `regions_uri`; every object is a carcinoma
region, the presets carry no class; objects are polygons, rectangles or freeform-tool
 multipolygons (with erased holes), see `region_geometry`), every nucleus of the 
`nuclei_batch_name` segmentation is labelled `annot_label = 1` when at least
`min_inside_fraction` of its polygon vertices lie inside the union of the regions 
(0.75; 1.0 would be the rule of the training labels, `overlap_threshold: 1.0`), else 0.

The regions are not exhaustive: 0 means "not inside a pathologist's region", not "benign".
The labels are meant for `data/supervision=annotation_sparse` only, which on a positive slide
supervises the label-1 nuclei and ignores all the others (`SparseAnnotationNucleiSupervision`);
negative test slides (normal / inflammation, low grade) need no label file, every nucleus of an
`is_carcinoma = False` slide being supervised as 0.

Writes `<output_path>/<slide_id>.parquet` (slide_id, id, annot_label) per slide and
`<output_path>/_summary.csv` (per slide: nuclei, nuclei inside the regions, regions) and the
`_SPARSE_ONLY` marker, with which any other supervision mode refuses the directory. The
summary and the labels hold slide ids, so only counts are logged.
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
import shapely
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import with_cli_args

from nuclei_graph.data.supervision import SPARSE_ONLY_MARKER


def _ring(points: list[dict]) -> shapely.Geometry:
    """One ring as a valid areal geometry; empty for a ring with no area."""
    if len(points) < 3:
        return shapely.Polygon()
    valid = shapely.make_valid(shapely.Polygon([(p["x"], p["y"]) for p in points]))
    # make_valid turns a zero-area ring into lines/points: keep only the areal parts
    parts = [g for g in shapely.get_parts(valid) if g.geom_type == "Polygon"]
    return shapely.union_all(parts) if parts else shapely.Polygon()


def region_geometry(obj: dict) -> shapely.Geometry:
    """One xOpat object as the area the viewer fills.

    `polygon`: a flat list of points. `multipolygon` (type `path`, what the freeform tool
    makes after erasing): a list of rings, filled with the even-odd rule (xOpat's
    Multipolygon factory sets `fillRule = "evenodd"`), so a ring inside another is a hole.
    Rings of fewer than 3 points (left over by the eraser) have no area and are dropped.
    `rect`: `left`, `top` (top-left corner, Fabric's default origin), `width`, `height`, the
    only geometry the Rect factory exports (its scale is reset to 1 after a resize).
    """
    if obj["factoryID"] == "polygon":
        return _ring(obj["points"])
    if obj["factoryID"] == "rect":
        x, y = obj["left"], obj["top"]
        return shapely.box(x, y, x + obj["width"], y + obj["height"])
    if obj["factoryID"] == "multipolygon":
        area = shapely.Polygon()
        for points in obj["points"]:
            area = shapely.symmetric_difference(area, _ring(points))  # even-odd
        return area
    raise ValueError(f"unsupported annotation object {obj['factoryID']!r}")


def load_regions(path: Path) -> shapely.Geometry:
    """The union of the slide's pathologist regions (level-0 px)."""
    objects = json.loads(path.read_text())["objects"]
    assert objects, f"{path.name}: no regions"
    regions = shapely.union_all([region_geometry(o) for o in objects])
    assert regions.area > 0, f"{path.name}: regions have no area"
    return regions


def label_slide(
    slide_id: str,
    regions_dir: Path,
    nuclei_dir: Path,
    output_dir: Path,
    label: str,
    min_inside_fraction: float,
) -> dict:
    region_path = regions_dir / f"{slide_id}.tiff.json"
    regions = load_regions(region_path)
    shapely.prepare(regions)
    nuclei = pd.read_parquet(
        nuclei_dir / f"slide_id={slide_id}", columns=["id", "polygon"]
    ).sort_values("id", ignore_index=True)
    vertices = np.stack(nuclei["polygon"].to_numpy()).reshape(len(nuclei), -1, 2)
    inside = shapely.contains_xy(
        regions, vertices[..., 0].ravel(), vertices[..., 1].ravel()
    ).reshape(vertices.shape[:2])
    out = pd.DataFrame(
        {
            "slide_id": slide_id,
            "id": nuclei["id"],
            label: (inside.mean(axis=1) >= min_inside_fraction).astype(np.int64),
        }
    )
    out.to_parquet(output_dir / f"{slide_id}.parquet", index=False)
    return {
        "slide_id": slide_id,
        "nuclei": len(out),
        "inside": int(out[label].sum()),
        "regions": len(json.loads(region_path.read_text())["objects"]),
    }


@with_cli_args(["+preprocessing=reannotation_labels_icaird_cervix"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
def main(config: DictConfig) -> None:
    mlflow.set_tracking_uri(config.regions_tracking_uri)
    regions_dir = Path(download_artifacts(config.regions_uri))
    slide_ids = sorted(
        p.name.removesuffix(".tiff.json") for p in regions_dir.glob("*.tiff.json")
    )
    assert slide_ids, "no region files"
    output_dir = Path(config.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 0 = unlabelled, not benign: only annotation_sparse may read this directory
    (output_dir / SPARSE_ONLY_MARKER).write_text(
        "0 = outside the pathologist's regions (unlabelled); use annotation_sparse only\n"
    )
    nuclei_dir = Path(config.nuclei_path) / config.nuclei_batch_name

    with ThreadPoolExecutor(config.max_workers) as pool:
        rows = list(
            pool.map(
                lambda s: label_slide(
                    s,
                    regions_dir,
                    nuclei_dir,
                    output_dir,
                    config.label_column,
                    config.min_inside_fraction,
                ),
                slide_ids,
            )
        )
    summary = pd.DataFrame(rows)
    assert (summary["inside"] > 0).all(), (
        "a slide has no nucleus inside its regions (coordinate frame?)"
    )
    summary.to_csv(output_dir / "_summary.csv", index=False)
    # the regions folder keeps changing while the pathologist works: record what was read
    region_files = pd.DataFrame(
        {
            "file": [f"{s}.tiff.json" for s in slide_ids],
            "sha256": [
                hashlib.sha256((regions_dir / f"{s}.tiff.json").read_bytes()).hexdigest()
                for s in slide_ids
            ],
        }
    )
    counts = {
        "slides": len(summary),
        "nuclei": int(summary["nuclei"].sum()),
        "nuclei_inside_regions": int(summary["inside"].sum()),
        "regions": int(summary["regions"].sum()),
    }
    print(json.dumps(counts, indent=1))

    if not config.log_mlflow:
        print(f"log_mlflow=false: labels in {output_dir}")
        return
    mlflow.set_experiment(config.metadata.experiment_name)
    with mlflow.start_run(run_name=config.metadata.run_name):
        mlflow.set_tag("mlflow.note.content", config.metadata.description)
        mlflow.log_params(
            {
                "regions_uri": config.regions_uri,
                "nuclei_batch_name": config.nuclei_batch_name,
                "rule": "annot_label = 1 iff >= min_inside_fraction of the polygon vertices "
                "inside the regions' union",
                "min_inside_fraction": config.min_inside_fraction,
                "output_path": str(output_dir),
            }
        )
        mlflow.log_metrics(counts)
        mlflow.log_table(region_files, "region_files.json")
        mlflow.log_text(OmegaConf.to_yaml(config, resolve=True), "config-resolved.yaml")


if __name__ == "__main__":
    main()
