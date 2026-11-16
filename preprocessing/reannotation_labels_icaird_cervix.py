"""Test labels of the iCAIRD cervix slides re-annotated by a pathologist.

For each positive test slide with the pathologist's regions (xOpat export, one
`annotations/<slide_id>.tiff.json` per slide in `regions_uri`; every object is a carcinoma
region, the presets carry no class), every nucleus of the `nuclei_batch_name` segmentation is
labelled `annot_label = 1` when all its polygon vertices lie inside the union of the regions
(the rule of the training labels, `overlap_threshold: 1.0`), else 0.

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


def load_regions(path: Path) -> shapely.Geometry:
    """The union of the slide's pathologist regions (level-0 px)."""
    objects = json.loads(path.read_text())["objects"]
    assert objects, f"{path.name}: no regions"
    assert all(o["type"] == "polygon" for o in objects), f"{path.name}: non-polygon"
    polygons = [
        shapely.make_valid(shapely.Polygon([(p["x"], p["y"]) for p in o["points"]]))
        for o in objects
    ]
    return shapely.union_all(polygons)


def label_slide(
    slide_id: str, regions_dir: Path, nuclei_dir: Path, output_dir: Path, label: str
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
            label: inside.all(axis=1).astype(np.int64),
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
                    s, regions_dir, nuclei_dir, output_dir, config.label_column
                ),
                slide_ids,
            )
        )
    summary = pd.DataFrame(rows)
    assert (summary["inside"] > 0).all(), (
        "a slide has no nucleus inside its regions (coordinate frame?)"
    )
    summary.to_csv(output_dir / "_summary.csv", index=False)
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
                "rule": "annot_label = 1 iff every polygon vertex inside the regions' union",
                "output_path": str(output_dir),
            }
        )
        mlflow.log_metrics(counts)
        mlflow.log_text(OmegaConf.to_yaml(config, resolve=True), "config-resolved.yaml")


if __name__ == "__main__":
    main()
