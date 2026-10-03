# credits: https://github.com/RationAI/carcinoma-binary-classification-methods/blob/master/preprocessing/masks/quality_control_v2.py

import asyncio
import re
import shutil
from collections.abc import Generator
from pathlib import Path
from typing import TypedDict

import hydra
import pandas as pd
import rationai
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from rationai.types import SlideCheckConfig
from tqdm.asyncio import tqdm


class QCParameters(TypedDict):
    mask_level: int
    sample_level: int
    check_residual: bool
    check_folding: bool
    check_focus: bool
    wb_correction: bool


def get_qc_masks(qc_parameters: QCParameters) -> Generator[tuple[str, str], None, None]:
    if qc_parameters["check_focus"]:
        yield ("Piqe/blur_score_coverage", "blur_per_tile")
        yield ("Piqe/blur_score_per_pixel", "blur_per_pixel")

    if qc_parameters["check_residual"]:
        yield ("ResidualArtifactsAndCoverage/artifacts_coverage", "residual_per_tile")
        yield ("ResidualArtifactsAndCoverage/artifacts_per_pixel", "residual_per_pixel")

    if qc_parameters["check_folding"]:
        yield ("FoldingFunction/folding_per_pixel", "folding_per_pixel")


def organize_masks(output_path: Path, subdir: str, current_subdir: str) -> None:
    prefix_dir = output_path / subdir
    prefix_dir.mkdir(parents=True, exist_ok=True)

    current_dir = output_path / current_subdir

    for file in list(current_dir.glob("*.tiff")):
        destination = prefix_dir / file.name
        file.rename(destination)


def failed_slides_from_log(log_uri: str) -> list[str]:
    """Slide paths listed in a `qc_errors.log` written by a previous run."""
    log_path = Path(download_artifacts(log_uri))
    pattern = re.compile(r"^Failed to process (.+?): ")
    return [
        match.group(1)
        for line in log_path.read_text().splitlines()
        if (match := pattern.match(line))
    ]


async def qc_main(
    output_path: Path,
    slides: list[str],
    logger: MLFlowLogger,
    request_timeout: int,
    max_concurrent: int,
    qc_parameters: QCParameters,
    base_url: str,
    artifact_path: str | None = None,
) -> None:
    async with rationai.AsyncClient(qc_base_url=base_url) as client:  # type: ignore[attr-defined]
        async for result in tqdm(
            client.qc.check_slides(
                slides,
                output_path,
                config=SlideCheckConfig(**qc_parameters),
                timeout=request_timeout,
                max_concurrent=max_concurrent,
            ),
            total=len(slides),
        ):
            if not result.success:
                with open(output_path / "qc_errors.log", "a") as log_file:  # noqa: ASYNC230
                    log_file.write(
                        f"Failed to process {result.wsi_path}: {result.error}\n"
                    )

        for prefix, artifact_name in get_qc_masks(qc_parameters):
            organize_masks(Path(output_path), artifact_name, prefix)

        csvs = list(Path(output_path).rglob("*.csv"))
        if len(csvs) > 1: 
            pd.concat([pd.read_csv(f) for f in csvs]).to_csv(
                Path(output_path, "qc_metrics.csv"), index=False
            )
            for f in csvs:
                f.unlink()

        logger.log_artifacts(local_dir=str(output_path), artifact_path=artifact_path)


@with_cli_args(["+preprocessing=quality_control"])
@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    dataset = pd.read_csv(download_artifacts(config.metadata_uri))

    slides = dataset["slide_path"].to_list()
    if config.get("retry_errors_uri"):
        failed = set(failed_slides_from_log(config.retry_errors_uri))
        slides = [slide for slide in slides if slide in failed]
        print(f"Retrying {len(slides)} slides from {config.retry_errors_uri}")

    output_path = Path(config.output_path)
    if output_path.exists():
        shutil.rmtree(str(output_path))

    output_path.mkdir(parents=True, exist_ok=True)

    asyncio.run(
        qc_main(
            output_path=output_path,
            slides=slides,
            logger=logger,
            request_timeout=config.request_timeout,
            max_concurrent=config.max_concurrent,
            qc_parameters=config.qc_parameters,
            base_url=config.base_url,
            artifact_path=config.get("artifact_path"),
        )
    )


if __name__ == "__main__":
    main()