"""Checks and statistics of the extracted activations (checkpoint after step 2).

Per split: slides, nuclei, nuclei per slide, file sizes, failed slides (missing in the
callback summary, or with any all-zero row of h) and a sanity check on `sample_size`
random nuclei: max |class_head(h) - logits|, with h as stored (float16) and the head
evaluated in float64. The runtime is read from the extraction's MLflow run.
"""

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from nuclei_graph.mlflow_utils import tag_parent_run
from nuclei_graph.xai.activations import load_activations
from nuclei_graph.xai.metrics import LinearHead
from nuclei_graph.xai.runtime import setup_precision


def extraction_runtime(parent_run_id: str, run_name: str) -> tuple[str, float]:
    runs = MlflowClient().search_runs(
        experiment_ids=[MlflowClient().get_run(parent_run_id).info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}' "
        f"and attributes.run_name = '{run_name}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    assert runs, f"No finished run '{run_name}'"
    info = runs[0].info
    return info.run_id, (info.end_time - info.start_time) / 1000


def zero_row_slides(h: np.ndarray, slides: pd.DataFrame) -> list[str]:
    """Slides with at least one all-zero row of h (never written)."""
    failed = []
    for slide in slides.itertuples(index=False):
        rows = np.asarray(h[slide.offset : slide.offset + slide.n_nuclei])
        if (~rows.any(axis=1)).any():
            failed.append(slide.slide_id)
    return failed


@with_cli_args(["+xai=check_activations"])
@hydra.main(config_path="../configs", config_name="xai", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)
    for key, value in setup_precision(config.matmul_precision).items():
        logger.experiment.set_tag(logger.run_id, key, value)

    activations_dir = Path(config.activations_dir)
    head = LinearHead.from_npz(str(activations_dir / "head.npz"))
    slides = pd.read_parquet(Path(config.out_dir) / "slides.parquet")
    summary = json.loads((activations_dir / "activations_summary.json").read_text())
    run_id, runtime = extraction_runtime(
        config.parent_run_id, config.extraction_run_name
    )
    rng = np.random.default_rng(config.seed)

    report: dict = {
        "extraction_run_id": run_id,
        "runtime_s": runtime,
        "callback_summary": {k: v for k, v in summary.items() if k != "missing"},
        "splits": {},
    }
    for split, split_slides in slides.groupby("split"):
        activations = load_activations(activations_dir, str(split))
        n_per_slide = split_slides["n_nuclei"]

        rows = np.sort(rng.choice(len(activations), config.sample_size, replace=False))
        deviation = np.abs(
            head(np.asarray(activations.h[rows], np.float64)) - activations.logits[rows]
        )
        failed = sorted(
            set(summary["missing"]) & set(split_slides["slide_id"])
            | set(zero_row_slides(activations.h, split_slides))
        )
        sizes = {
            name: (activations_dir / str(split) / name).stat().st_size
            for name in ("h.npy", "logits.npy", "index.parquet")
        }
        report["splits"][str(split)] = {
            "slides": len(split_slides),
            "nuclei": len(activations),
            "nuclei_per_slide": {
                "median": float(n_per_slide.median()),
                "min": int(n_per_slide.min()),
                "max": int(n_per_slide.max()),
            },
            "carcinoma_nuclei": int(activations.labels.sum()),
            "disk_bytes": sizes,
            "failed_slides": failed,
            "sanity_check": {
                "sample_size": config.sample_size,
                "max_abs_diff": float(deviation.max()),
                "mean_abs_diff": float(deviation.mean()),
                "atol": config.atol,
                "passed": bool(deviation.max() < config.atol),
            },
        }
        logger.log_metrics(
            {
                f"{split}/slides": len(split_slides),
                f"{split}/nuclei": len(activations),
                f"{split}/failed_slides": len(failed),
                f"{split}/sanity_max_abs_diff": float(deviation.max()),
                f"{split}/disk_gb": sum(sizes.values()) / 1e9,
            }
        )

    path = activations_dir / "check.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    logger.log_metrics({"extraction_runtime_s": runtime})
    logger.log_artifact(str(path), "activations")
    assert all(s["sanity_check"]["passed"] for s in report["splits"].values())


if __name__ == "__main__":
    main()
