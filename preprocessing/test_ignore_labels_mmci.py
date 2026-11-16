"""Test-time ignore labels of the MMCI TL test set: Another pathology and Exclude nuclei.

The final MMCI evaluation rule ignores, at test, every nucleus of an Another pathology or
Exclude region, matching the tile-level evaluation (the filtered tiles drop every tile that
touches either). Training is unchanged (Another pathology ignored through the pipeline's
`another_pathology` files, Exclude labelled 0).

Per test slide, over all its nuclei (`get_slide_data`, sorted by id):
    - `another_pathology`: `ignore_label = 1` in the pipeline's Another pathology ignore files
      (the files training uses);
    - `exclude`: centroid inside the slide's "Exclude" annotation polygons (ASAP XML next to the
      slide), the rule of `region_counts` in postprocessing/nucleus_results_mmci.py;
    - `ignore_label` = either.

A file `<output_path>/<slide_id>.parquet` (slide_id, id, another_pathology, exclude,
ignore_label) is written for every test slide with at least one ignored nucleus; it is the
`ignore_dir` of the MMCI nucleus-level test config and of the postprocessing defaults.
`_summary.csv` holds the per-slide counts.
"""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import shapely
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import with_cli_args
from ratiopath.parsers import ASAPParser
from shapely import MultiPolygon
from shapely.ops import unary_union

from nuclei_graph.data.datasets.tile.base import get_slide_data


def exclude_mask(xml: Path, centroids: np.ndarray) -> np.ndarray:
    """Centroids inside the slide's "Exclude" polygons."""
    if not xml.exists():
        return np.zeros(len(centroids), bool)
    g = MultiPolygon(list(ASAPParser(xml).get_polygons(part_of_group="Exclude")))
    if g.is_empty:
        return np.zeros(len(centroids), bool)
    g = unary_union(shapely.make_valid(g))
    return shapely.contains_xy(g, centroids[:, 0], centroids[:, 1])


def another_mask(path: Path, ids: np.ndarray) -> np.ndarray:
    if not path.exists():
        return np.zeros(len(ids), bool)
    d = pd.read_parquet(path, columns=["id", "ignore_label"]).set_index("id")
    assert set(d["ignore_label"].unique()) <= {0, 1}
    assert set(d.index) <= set(ids), f"{path.name}: ids without a nucleus"
    return d["ignore_label"].reindex(ids).fillna(0).to_numpy() == 1


@with_cli_args(["+preprocessing=test_ignore_labels_mmci"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
def main(config: DictConfig) -> None:
    meta = pd.read_parquet(download_artifacts(config.metadata_uri)).sort_values(
        "slide_id", ignore_index=True
    )
    out = Path(config.output_path)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in meta.itertuples():
        _, centroids, _, ids = get_slide_data(str(r.slide_nuclei_path))
        another = another_mask(
            Path(config.another_pathology_dir) / f"{r.slide_id}.parquet", ids
        )
        exclude = exclude_mask(
            Path(r.slide_path).with_suffix(config.annotation_suffix), centroids
        )
        ignore = another | exclude
        rows.append(
            {
                "slide_id": r.slide_id,
                "nuclei": len(ids),
                "another_pathology": int(another.sum()),
                "exclude": int(exclude.sum()),
                "both": int((another & exclude).sum()),
                "ignored": int(ignore.sum()),
            }
        )
        path = out / f"{r.slide_id}.parquet"
        if ignore.any():
            pd.DataFrame(
                {
                    "slide_id": r.slide_id,
                    "id": ids,
                    "another_pathology": another.astype(np.int8),
                    "exclude": exclude.astype(np.int8),
                    "ignore_label": ignore.astype(np.int8),
                }
            ).to_parquet(path, index=False)
        elif path.exists():
            path.unlink()  # a stale file of an earlier rule
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "_summary.csv", index=False)
    with_any = summary[summary["ignored"] > 0]
    print(
        f"{len(with_any)} of {len(summary)} test slides with ignored nuclei: "
        f"Another pathology {summary['another_pathology'].sum():,} "
        f"({(summary['another_pathology'] > 0).sum()} slides), Exclude "
        f"{summary['exclude'].sum():,} ({(summary['exclude'] > 0).sum()} slides), both "
        f"{summary['both'].sum():,}, ignored {summary['ignored'].sum():,}; files in {out}"
    )


if __name__ == "__main__":
    main()
