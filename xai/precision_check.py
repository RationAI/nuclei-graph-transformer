"""Test logits in full fp32 (no TF32) next to the stored TF32 logits.

Runs the same whole-slide inference as the activation extraction on the test slides,
with `torch.set_float32_matmul_precision(matmul_precision)` instead of the repo's
"high" (TF32), into `precision_check/`. It reports AUROC/AUPRC of both against the
dense labels and the distribution of per-nucleus |Δlogit|. The stored arrays are not
changed; the fp32 `h.npy` is deleted after the comparison.
"""

from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from lightning import seed_everything
from omegaconf import DictConfig
from rationai.mlkit import Trainer, autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

import nuclei_graph.__main__  # noqa: F401  (resolvers)
from nuclei_graph.xai.metrics import classification_metrics
from nuclei_graph.xai.runtime import setup_precision


@with_cli_args(
    [
        "~experiment/modeling/training",
        "+experiment/modeling/prediction=nuclei_level/radboud_activations_fp32",
    ]
)
@hydra.main(config_path="../configs", config_name="ml", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    for key, value in setup_precision(config.matmul_precision).items():
        logger.experiment.set_tag(logger.run_id, key, value)
    seed_everything(config.seed, workers=True)

    out_dir = Path(config.xai_out_dir)
    check_dir = Path(config.precision_check_dir)
    check_dir.mkdir(parents=True, exist_ok=True)
    slides = pd.read_parquet(out_dir / "slides.parquet")
    slides[slides["split"] == "test"].to_parquet(check_dir / "slides_test.parquet")

    data = instantiate(config.data, _recursive_=False)
    model = instantiate(config.model)
    trainer = instantiate(config.trainer, _target_=Trainer, logger=logger)
    trainer.predict(model, datamodule=data, ckpt_path=config.checkpoint)

    tf32 = np.load(out_dir / "activations" / "test" / "logits.npy")
    fp32 = np.load(check_dir / "test" / "logits.npy")
    labels = pd.read_parquet(
        out_dir / "activations" / "test" / "index.parquet", columns=["label"]
    )["label"].to_numpy()
    delta = np.abs(fp32.astype(np.float64) - tf32)

    report = pd.DataFrame(
        [
            {"precision": "tf32 (stored)", **classification_metrics(labels, tf32)},
            {
                "precision": config.matmul_precision,
                **classification_metrics(labels, fp32),
            },
        ]
    )
    stats = {
        "delta_logit_median": float(np.median(delta)),
        "delta_logit_p99": float(np.quantile(delta, 0.99)),
        "delta_logit_max": float(delta.max()),
        "delta_logit_mean": float(delta.mean()),
        "sign_flips": int((np.sign(fp32) != np.sign(tf32)).sum()),
    }
    report.to_csv(check_dir / "metrics.csv", index=False)
    pd.Series(stats).to_csv(check_dir / "delta_logit.csv", header=["value"])
    print(report.to_string(index=False))
    print(stats)
    logger.log_metrics(stats)
    for row in report.itertuples(index=False):
        tag = "tf32" if row.precision.startswith("tf32") else "fp32"
        logger.log_metrics({f"{tag}/AUROC": row.AUROC, f"{tag}/AUPRC": row.AUPRC})
    logger.log_artifact(str(check_dir / "metrics.csv"), "precision_check")
    logger.log_artifact(str(check_dir / "delta_logit.csv"), "precision_check")
    (check_dir / "test" / "h.npy").unlink()  # 7.5 GB, not needed


if __name__ == "__main__":
    main()
