"""Script for mapping the slide-level metadata of iCAIRD cervix to its nuclei.

One table per split of the dataset's index (train, valid, test), logged as
`icaird_cervix/slides_mapping_<split>.parquet` and as MLflow dataset inputs. A slide is
positive (`is_carcinoma`) when its category is high grade or malignant, the classes merged
into the carcinoma annotation masks the nuclei labels come from.

Train, valid and test hold only the slides of `categories`: low grade is used nowhere (no
training, validation or evaluation), and no table holds it. The test table is the slides
re-annotated by a pathologist (positive) plus every test slide of `test_slides.categories`
(negative); every positive test slide must be re-annotated.

Left out, and listed with the reasons in `excluded_slides.csv`:
    - slides whose category is not in `categories` (low grade);
    - slides listed in `exclude_slides_uris` (the blur exclusion of the QC);
    - slides with no nuclei;
    - test slides that are neither in `test_slides.slides_uri` (the slides re-annotated by a
      pathologist) nor of a category in `test_slides.categories` (the negatives kept whole).
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import mlflow
import pandas as pd
import pyarrow.parquet as pq
from mlflow.artifacts import download_artifacts
from mlflow.data import pandas_dataset
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.openslide import OpenSlide


POSITIVE_CATEGORIES = ("high_grade", "malignant")


def get_mpp(slide_path: str) -> tuple[float, float]:
    with OpenSlide(slide_path) as slide:
        return slide.slide_resolution(level=0)


def count_nuclei(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(pq.ParquetFile(f).metadata.num_rows for f in path.glob("*.parquet"))


def log_table(df: pd.DataFrame, name: str, logger: MLFlowLogger) -> None:
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir, f"{name}.parquet")
        df.to_parquet(path, index=False)
        logger.log_artifact(str(path), artifact_path="icaird_cervix")
        mlflow.log_input(
            pandas_dataset.from_pandas(df, name=name), context=f"icaird_cervix/{name}"
        )


@with_cli_args(["+preprocessing=metadata_mapping/icaird_cervix"])
@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    slides = pd.read_csv(download_artifacts(config.metadata_uri))
    excluded = pd.concat(
        [
            pd.read_csv(download_artifacts(uri))
            for uri in config.exclude_slides_uris or []
        ]
        or [pd.DataFrame(columns=["slide_path"])],
        ignore_index=True,
    )

    nuclei_dir = Path(config.nuclei_path) / config.nuclei_batch_name
    nuclei_paths = slides["slide_id"].map(lambda id: nuclei_dir / f"slide_id={id}")
    with ThreadPoolExecutor(16) as pool:
        mpp_x, mpp_y = zip(*pool.map(get_mpp, slides["slide_path"]), strict=True)
        nuclei_counts = list(pool.map(count_nuclei, nuclei_paths))
    map_df = pd.DataFrame(
        {
            "slide_id": slides["slide_id"],
            "slide_path": slides["slide_path"],
            "slide_nuclei_path": nuclei_paths.map(str),
            "nuclei_count": pd.Series(nuclei_counts, index=slides.index).astype(
                "Int64"
            ),
            "is_carcinoma": slides["category"].isin(POSITIVE_CATEGORIES),
            "category": slides["category"],
            "subcategory": slides["subcategory"],
            "mpp_x": mpp_x,
            "mpp_y": mpp_y,
            "split": slides["split"],
        }
    )

    test_slides = set(
        pd.read_csv(download_artifacts(config.test_slides.slides_uri))["slide_id"]
    )
    reasons = {
        f"category not used ({', '.join(config.categories)})": ~map_df["category"].isin(
            config.categories
        ),
        "qc: blur": map_df["slide_path"].isin(excluded["slide_path"]),
        "no nuclei": map_df["nuclei_count"] == 0,
        "test: not re-annotated and not a kept negative": (map_df["split"] == "test")
        & ~map_df["slide_id"].isin(test_slides)
        & ~map_df["category"].isin(config.test_slides.categories),
    }
    reason = pd.Series("", index=map_df.index)
    for name, mask in reasons.items():
        reason[mask] = reason[mask].map(lambda r, n=name: f"{r}; {n}" if r else n)

    dropped = map_df.loc[reason != "", ["slide_id", "split", "category"]].assign(
        reason=reason[reason != ""]
    )
    with TemporaryDirectory() as temp_dir:
        path = Path(temp_dir, "excluded_slides.csv")
        dropped.to_csv(path, index=False)
        logger.log_artifact(str(path), artifact_path="icaird_cervix")
    missing = test_slides - set(map_df.loc[reason == "", "slide_id"])
    if missing:
        raise ValueError(f"re-annotated test slides left out: {sorted(missing)}")

    kept = map_df[reason == ""]
    assert kept["category"].isin(config.categories).all(), (
        "a category outside `categories` (low grade) in train, valid or test"
    )
    # positive test slides are labelled only inside the pathologist's regions (sparse labels of
    # preprocessing/reannotation_labels_icaird_cervix.py): every one must be re-annotated
    test_positive = set(
        kept.loc[(kept["split"] == "test") & kept["is_carcinoma"], "slide_id"]
    )
    assert test_positive == test_slides, (
        "positive test slides != the re-annotated slides"
    )
    for split_name in ["train", "valid", "test"]:
        split_df = kept[kept["split"] == split_name].drop(columns=["split"])
        log_table(split_df, f"slides_mapping_{split_name}", logger)


if __name__ == "__main__":
    main()
