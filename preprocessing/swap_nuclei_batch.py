"""Replaces the nuclei of selected slides with a re-segmented batch.

For every slide in the metadata CSV, `<target_batch>/slide_id=<slide>` (the batch the
rest of the pipeline reads) is moved to `<backup_batch>/` and
`<source_batch>/slide_id=<slide>` takes its place. All slides are checked before
anything is moved; with `dry_run` nothing is moved at all. Undo: move the folders back
from `backup_batch` by hand (the logged `swaps.csv` lists them).

Nucleus ids are derived from the slide, tile and index within the tile, so a
re-segmentation reuses ids for different nuclei: every per-nucleus table of the swapped
slides (labels, predictions, ...) has to be regenerated afterwards.
"""

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import pandas as pd
import pyarrow.parquet as pq
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger


def count_nuclei(slide_dir: Path) -> int:
    return sum(pq.ParquetFile(f).metadata.num_rows for f in slide_dir.glob("*.parquet"))


@with_cli_args(["+preprocessing=swap_nuclei_batch"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    nuclei_path = Path(config.nuclei_path)
    slide_ids = pd.read_csv(download_artifacts(config.metadata_uri))["slide_id"]

    swaps = []
    for slide_id in slide_ids:
        folder = f"slide_id={slide_id}"
        source = nuclei_path / config.source_batch / folder
        target = nuclei_path / config.target_batch / folder
        backup = nuclei_path / config.backup_batch / folder
        if not source.is_dir() or count_nuclei(source) == 0:
            raise FileNotFoundError(f"{source}: no re-segmented nuclei")
        if not target.is_dir():
            raise FileNotFoundError(f"{target}: no nuclei to replace")
        if backup.exists():
            raise FileExistsError(f"{backup}: already swapped?")
        swaps.append(
            {
                "slide_id": slide_id,
                "source": str(source),
                "target": str(target),
                "backup": str(backup),
                "nuclei_before": count_nuclei(target),
                "nuclei_after": count_nuclei(source),
            }
        )
    swaps_df = pd.DataFrame(swaps)
    swaps_df["gain"] = (swaps_df.nuclei_after / swaps_df.nuclei_before).round(3)
    print(swaps_df[["slide_id", "nuclei_before", "nuclei_after", "gain"]].to_string())

    if not config.dry_run:
        for swap in swaps:
            Path(swap["backup"]).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(swap["target"], swap["backup"])
            shutil.move(swap["source"], swap["target"])

    with TemporaryDirectory() as output_dir:
        swaps_df.to_csv(Path(output_dir) / "swaps.csv", index=False)
        logger.log_artifacts(local_dir=output_dir)


if __name__ == "__main__":
    main()
