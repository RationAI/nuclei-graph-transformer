"""Ignore labels of the MMCI TL slides: Another pathology and Exclude nuclei, in every split.

Nuclei of Another pathology and Exclude regions are left out of the supervision of training,
validation and test (`ignore_dir`), matching the tile level, where the filtered tiles drop every
tile touching either. They stay in the model input as context; only their own loss and metrics
are dropped.

Per slide of the `metadata_uris` tables (train, which holds the validation split, and test), over
all its nuclei (sorted by id, as `get_nuclei` and `get_slide_data`):
    - `another_pathology`: `ignore_label = 1` in the pipeline's Another pathology ignore files;
    - `exclude`: centroid inside the slide's "Exclude" annotation polygons (ASAP XML next to the
      slide), the rule of `region_counts` in postprocessing/nucleus_results_mmci.py;
    - `ignore_label` = either.

A file `<output_path>/<slide_id>.parquet` (slide_id, id, another_pathology, exclude,
ignore_label) is written for every slide with at least one ignored nucleus. `_summary.csv`
holds the per-slide counts, with the ignored nuclei the annotation labels mark 1. Exclude nuclei
must have none (the carcinoma annotation excludes the region), nor may the test set. A few
Another pathology nuclei of training slides do (25 in 5 slides, at the region border where the
pipeline's ignore files and the annotation labels disagree); they were already ignored by training.
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


def exclude_mask(xml: Path, centroids: np.ndarray) -> np.ndarray:
    """Centroids inside the slide's "Exclude" polygons."""
    if not xml.exists():
        return np.zeros(len(centroids), bool)
    g = MultiPolygon(list(ASAPParser(xml).get_polygons(part_of_group="Exclude")))
    if g.is_empty:
        return np.zeros(len(centroids), bool)
    g = unary_union(shapely.make_valid(g))
    return shapely.contains_xy(g, centroids[:, 0], centroids[:, 1])


def id_mask(path: Path, ids: np.ndarray, column: str) -> np.ndarray:
    """Nuclei with `column` = 1 in a per-slide file (none if the file is missing)."""
    if not path.exists():
        return np.zeros(len(ids), bool)
    d = pd.read_parquet(path, columns=["id", column]).set_index("id")[column]
    assert set(d.unique()) <= {0, 1}
    assert set(d.index) <= set(ids), f"{path.name}: ids without a nucleus"
    return d.reindex(ids).fillna(0).to_numpy() == 1


@with_cli_args(["+preprocessing=ignore_labels_mmci"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
def main(config: DictConfig) -> None:
    meta = pd.concat(
        [
            pd.read_parquet(download_artifacts(uri)).assign(split=split)
            for split, uri in config.metadata_uris.items()
        ],
        ignore_index=True,
    ).sort_values("slide_id", ignore_index=True)
    assert meta["slide_id"].is_unique, "a slide in two tables"
    out = Path(config.output_path)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in meta.itertuples():
        nuclei = pd.read_parquet(r.slide_nuclei_path, columns=["id", "centroid"])
        nuclei = nuclei.sort_values("id", ignore_index=True)
        ids, centroids = nuclei["id"].to_numpy(), np.stack(nuclei["centroid"])
        another = id_mask(
            Path(config.another_pathology_dir) / f"{r.slide_id}.parquet",
            ids,
            "ignore_label",
        )
        exclude = exclude_mask(
            Path(r.slide_path).with_suffix(config.annotation_suffix), centroids
        )
        positive = id_mask(
            Path(config.supervision_dir) / f"{r.slide_id}.parquet",
            ids,
            config.label_column,
        )
        ignore = another | exclude
        rows.append(
            {
                "slide_id": r.slide_id,
                "split": r.split,
                "nuclei": len(ids),
                "another_pathology": int(another.sum()),
                "exclude": int(exclude.sum()),
                "both": int((another & exclude).sum()),
                "ignored": int(ignore.sum()),
                "ignored_labelled_1": int((ignore & positive).sum()),
                "another_pathology_labelled_1": int((another & positive).sum()),
                "exclude_labelled_1": int((exclude & positive).sum()),
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
    assert summary["exclude_labelled_1"].sum() == 0, "an Exclude nucleus is labelled 1"
    test = summary[summary["split"] == "test"]
    assert test["ignored_labelled_1"].sum() == 0, (
        "an ignored test nucleus is labelled 1"
    )
    for split, s in summary.groupby("split"):
        print(
            f"{split}: {(s['ignored'] > 0).sum()} of {len(s)} slides with ignored nuclei; "
            f"Another pathology {s['another_pathology'].sum():,} "
            f"({(s['another_pathology'] > 0).sum()} slides), Exclude {s['exclude'].sum():,} "
            f"({(s['exclude'] > 0).sum()} slides), both {s['both'].sum():,}, "
            f"ignored {s['ignored'].sum():,} (labelled 1: Another pathology "
            f"{s['another_pathology_labelled_1'].sum()}, Exclude {s['exclude_labelled_1'].sum()})"
        )
    print(f"files in {out}")


if __name__ == "__main__":
    main()
