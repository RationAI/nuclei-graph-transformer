r"""tab:res-nucleus: nucleus-level test results of the model on PANDA Radboud.

The model: shape embedding (efd), rotated values (``attention_variant: vrope``), k-NN
block-sparse attention with k = 24; three seeds. Inference outputs only: the per-nucleus test
predictions stored in the three evaluation runs (``<run>/artifacts/predictions``). No training,
no inference.

0. Runs. For each evaluation run: its training run (``mlflow.parentRunId``) must have the
   expected configuration (``expected_config``), a single checkpoint selected by
   ``checkpoint_monitor`` (the validation split of the train metadata; the test metadata is not
   read in fit/validate, see ``CropDataModule.setup``), and the evaluation must load that
   checkpoint at the training run's ``best/epoch``. Exactly three runs, or the script stops.
1. Labels as in the existing evaluation (``postprocessing.bootstrap_sources``): annotation
   supervision left-joined on (slide, nucleus), missing -> 0, every nucleus of an
   ``is_carcinoma = False`` slide -> 0. Radboud labels are dense (0/1), so no nucleus is ignored.
2. Metrics: AUPRC (torchmetrics ``binary_average_precision``) and AUROC (``binary_auroc``), exact,
   over all labelled test nuclei pooled; per seed, then the mean and SD over seeds (predictions
   are not averaged).
3. Bootstrap: the canonical slide resamples of Radboud test (``resamples``; stratified by
   positive / negative slide, B = 2,000; Q2-Q4 use the same file). Per resample, every seed's
   metric weights each nucleus by its slide's multiplicity, and the three seeds are averaged;
   percentile CIs. Replicates come from per-slide histograms of the logits of positive and
   negative nuclei on one uniform grid; the grid is refined until every seed's full-test-set
   metrics from the histograms match the exact torchmetrics values within ``histogram.tolerance``.

Outputs (``out_dir``, all logged to one MLflow run): ``nucleus_results.csv``, ``runs.csv``,
``checks.json``, ``report.md``, ``replicates.npz`` and the resamples file. Before logging, every
text output is checked for credentials in URLs and for slide ids (only ``runs.csv`` /
``checks.json`` hold run ids; the resamples file holds the public PANDA image ids it needs);
after logging, the run's tags are checked for credentials.
"""

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import yaml
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_average_precision,
)

from postprocessing.bootstrap_sources import (
    bin_logits,
    hist_metrics,
    label_predictions,
    load_labels,
    load_run,
    replicate_metrics,
    slide_histograms,
    to_logits,
)


METRICS = ("AUPRC", "AUROC")
CREDENTIAL_URL = re.compile(r"https?://[^/\s\"']*@")
CKPT_EPOCH = re.compile(r"/checkpoints/epoch=(\d+)-step=\d+/")
CODE = (
    "postprocessing/nucleus_results.py",
    "configs/postprocessing/nucleus_results.yaml",
)


# --------------------------------------------------------------------------
# 0. Runs
# --------------------------------------------------------------------------


def run_config(client: MlflowClient, run_id: str, tmp: Path) -> dict:
    (tmp / run_id).mkdir(parents=True, exist_ok=True)
    path = client.download_artifacts(
        run_id, "configs/config-resolved.yaml", str(tmp / run_id)
    )
    return yaml.safe_load(Path(path).read_text())


def model_config(cfg: dict) -> dict[str, str]:
    """The model configuration from a resolved run config (authoritative; MLflow params may be
    missing, e.g. ``attention_variant`` was not logged as a param by some training runs)."""
    net = cfg["model"]["net"]["config"]
    return {
        "embedding_mode": str(cfg["data"]["dataset"]["embedding_mode"]),
        "context": str(cfg["data"]["context"]),
        "k": str(cfg["data"]["k"]),
        "use_rope": str(net.get("use_rope", True)),  # default of the transformer config
        "attention_variant": str(net.get("attention_variant", "standard")),
    }


def check_runs(config: DictConfig, client: MlflowClient, tmp: Path) -> pd.DataFrame:
    fix = pd.Timestamp(config.block_mask_fix_date, tz="UTC")
    rows = []
    for seed, eval_id in enumerate(config.eval_runs, start=1):
        ev = client.get_run(eval_id)
        train_id = ev.data.tags["mlflow.parentRunId"]
        tr = client.get_run(train_id)
        ev_cfg, tr_cfg = run_config(client, eval_id, tmp), run_config(client, train_id, tmp)

        ckpt = ev_cfg["checkpoint"]
        epoch = int(CKPT_EPOCH.search(ckpt).group(1))
        monitors = [
            cb.get("monitor")
            for cb in tr_cfg["trainer"]["callbacks"]
            if isinstance(cb, dict) and cb.get("_target_", "").endswith("ModelCheckpoint")
        ]
        checkpoints = [f.path for f in client.list_artifacts(train_id, "checkpoints")]
        rows.append(
            {
                "seed": seed,
                "training_run": train_id,
                "eval_run": eval_id,
                **model_config(tr_cfg),
                **{f"eval_{k}": v for k, v in model_config(ev_cfg).items()},
                "params_not_logged": ",".join(
                    k for k in config.expected_config if k not in tr.data.params
                ),
                "params_disagreeing_with_config": ",".join(
                    k
                    for k in config.expected_config
                    if k in tr.data.params and tr.data.params[k] != model_config(tr_cfg)[k]
                ),
                "checkpoint_monitor": ",".join(map(str, monitors)),
                "training_split_size": tr_cfg["data"]["split_size"],
                "checkpoints_in_training_run": len(checkpoints),
                "checkpoint_run": ckpt.split("/37/")[1].split("/")[0],
                "checkpoint_epoch": epoch,
                "best_validation_epoch": int(tr.data.metrics["best/epoch"]),
                "best_validation_loss": tr.data.metrics.get("best/validation/loss"),
                "best_validation_AUPRC": tr.data.metrics.get("best/validation/AUPRC"),
                "last_logged_epoch": int(tr.data.metrics["epoch"]),
                "max_epochs": tr_cfg["trainer"]["max_epochs"],
                "training_status": tr.info.status,
                "training_resumed_from": (tr_cfg.get("checkpoint") or "").split("/37/")[-1].split("/")[0]
                or None,
                "training_end": pd.Timestamp(tr.info.end_time, unit="ms", tz="UTC")
                if tr.info.end_time
                else None,
                "trained_before_block_mask_fix": pd.Timestamp(
                    max(m.timestamp for m in client.get_metric_history(train_id, "epoch")),
                    unit="ms",
                    tz="UTC",
                )
                < fix,
                "evaluated_before_block_mask_fix": pd.Timestamp(
                    ev.info.start_time, unit="ms", tz="UTC"
                )
                < fix,
            }
        )
    runs = pd.DataFrame(rows)

    assert len(runs) == 3 and runs["training_run"].is_unique, "need exactly 3 distinct seeds"
    assert (runs["params_disagreeing_with_config"] == "").all(), "MLflow params contradict the config"
    for key, value in config.expected_config.items():
        assert (runs[key] == str(value)).all(), f"training {key} != {value}"
        assert (runs[f"eval_{key}"].astype(str) == str(value)).all(), f"evaluation {key} != {value}"
    assert (runs["checkpoint_monitor"] == config.checkpoint_monitor).all()
    assert (runs["training_split_size"] > 0).all(), "no validation split"
    assert (runs["checkpoints_in_training_run"] == 1).all()
    assert (runs["checkpoint_run"] == runs["training_run"]).all(), "evaluation loads another run's checkpoint"
    assert (runs["checkpoint_epoch"] == runs["best_validation_epoch"]).all(), (
        "evaluated checkpoint is not the best validation epoch"
    )
    return runs


# --------------------------------------------------------------------------
# 1-3. Labels, metrics, bootstrap
# --------------------------------------------------------------------------


def exact_metrics(y: NDArray, p: NDArray) -> dict[str, float]:
    preds, target = torch.from_numpy(p.astype(np.float32)), torch.from_numpy(y.astype(np.int64))
    return {
        "AUPRC": float(binary_average_precision(preds, target, thresholds=None)),
        "AUROC": float(binary_auroc(preds, target, thresholds=None)),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute(config: DictConfig, client: MlflowClient) -> tuple[Path, dict]:
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp"
    runs = check_runs(config, client, tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    print(runs.T.to_string())

    slides = pd.read_parquet(download_artifacts(config.metadata_uri))[
        ["slide_id", "is_carcinoma"]
    ].sort_values("slide_id", ignore_index=True)
    assert slides["slide_id"].is_unique
    slide_ids = slides["slide_id"].tolist()
    carcinoma = slides["is_carcinoma"].to_numpy(bool)
    labels = load_labels(slides, Path(config.supervision_dir), config.label_column)
    assert set(np.unique(labels["label_sup"])) <= {0, 1}, "Radboud labels must be dense 0/1"

    frames = []
    for eval_id in config.eval_runs:
        df, info = load_run(eval_id, slide_ids, Path(config.cache_dir), config.predictions_artifact)
        assert info["test_slides_without_predictions"] == 0
        assert info["files_outside_test_metadata"] == 0
        frames.append(df)
    ref = frames[0][["slide", "key"]]
    for f in frames[1:]:
        assert np.array_equal(f["slide"].to_numpy(), ref["slide"].to_numpy())
        assert np.array_equal(f["key"].to_numpy(), ref["key"].to_numpy()), "seeds cover different nuclei"
    y, unmatched = label_predictions(ref, labels, carcinoma)
    slide = ref["slide"].to_numpy()
    probs = np.stack([f["p"].to_numpy() for f in frames])
    n_slides = len(slides)

    positives_per_slide = np.bincount(slide, weights=(y == 1), minlength=n_slides)
    positive = positives_per_slide > 0
    counts = {
        "test_slides": n_slides,
        "positive_slides": int(positive.sum()),
        "negative_slides": int((~positive).sum()),
        "patients": None,  # not applicable (PANDA)
        "labeled_nuclei": len(y),
        "positive_nuclei": int((y == 1).sum()),
        "negative_nuclei": int((y == 0).sum()),
        "ignored_nuclei": 0,  # dense supervision; asserted above
        "prevalence": float(np.mean(y == 1)),
        "carcinoma_slide_nuclei_without_supervision_row": unmatched,
    }
    assert np.bincount(slide, minlength=n_slides).min() > 0, "a test slide has no nuclei"

    exact = [exact_metrics(y, p) for p in probs]
    for i, e in enumerate(exact):
        runs.loc[i, [f"{m}_exact" for m in METRICS]] = [e[m] for m in METRICS]
    print("exact (torchmetrics):", exact)

    # canonical resamples -----------------------------------------------------
    resamples = Path(config.resamples)
    r = np.load(resamples)
    assert int(r["seed"]) == int(config.resamples_seed)
    assert list(r["slide_ids"]) == slide_ids, "resamples are for other slides or another order"
    assert np.array_equal(r["positive"], positive), "resample strata differ from the labels"
    weights = r["multiplicities"].astype(np.float64)

    # histograms, validated against the exact metrics --------------------------
    logits = [to_logits(p) for p in probs]
    finite = np.concatenate([lg[np.isfinite(lg)] for lg in logits])
    lo, hi = float(finite.min()), float(finite.max())
    del finite
    n_bins = int(config.histogram.n_bins)
    while True:
        hists, errors = [], []
        for lg, e in zip(logits, exact, strict=True):
            pos, neg = slide_histograms(slide, bin_logits(lg, lo, hi, n_bins), y, n_slides, n_bins)
            full = hist_metrics(pos.sum(0), neg.sum(0))
            errors.append({m: float(full[m] - e[m]) for m in METRICS})
            hists.append((pos, neg))
        max_err = max(abs(v) for err in errors for v in err.values())
        print(f"histogram grid {n_bins} bins over logit [{lo:.3f}, {hi:.3f}]: max |error| {max_err:.2e}")
        if max_err <= config.histogram.tolerance or n_bins >= config.histogram.max_bins:
            break
        n_bins *= 2
    assert max_err <= config.histogram.tolerance, "histogram grid too coarse at max_bins"
    for i, err in enumerate(errors):
        runs.loc[i, [f"{m}_hist_minus_exact" for m in METRICS]] = [err[m] for m in METRICS]

    rep = {m: np.empty((len(probs), len(weights))) for m in METRICS}
    for i, (pos, neg) in enumerate(hists):
        for m, v in replicate_metrics(pos, neg, weights).items():
            rep[m][i] = v
    del hists
    np.savez_compressed(
        out / "replicates.npz", eval_runs=np.array(list(config.eval_runs)), **rep
    )

    # results -----------------------------------------------------------------
    q = (1 - config.ci_level) / 2 * 100
    row = {"dataset": config.dataset_label, **{k: counts[k] for k in (
        "test_slides", "patients", "labeled_nuclei", "positive_nuclei", "ignored_nuclei", "prevalence")}}
    for m in METRICS:
        per_seed = np.array([e[m] for e in exact])
        lo_ci, hi_ci = np.percentile(rep[m].mean(0), [q, 100 - q])
        row |= {
            m: float(per_seed.mean()),
            f"{m}_ci_lo": float(lo_ci),
            f"{m}_ci_hi": float(hi_ci),
            **{f"{m}_seed_{k + 1}": float(v) for k, v in enumerate(per_seed)},
            f"{m}_seed_sd": float(per_seed.std(ddof=1)),
        }
    results = pd.DataFrame([row])
    results.to_csv(out / "nucleus_results.csv", index=False)
    runs.to_csv(out / "runs.csv", index=False)
    shutil.copyfile(resamples, out / resamples.name)
    assert sha256(out / resamples.name) == sha256(resamples)

    checks = {
        "counts": counts,
        "labels": "annot_label left-joined on (slide_id, id), missing -> 0, is_carcinoma = False -> 0; "
        "dense labels, no nucleus ignored",
        "positive_slide_rule": "at least one nucleus labelled carcinoma",
        "metrics": "torchmetrics binary_average_precision / binary_auroc (thresholds=None), exact, "
        "per seed over all labelled test nuclei pooled; seed mean and SD (ddof=1)",
        "resamples": {
            "file": resamples.name,
            "source": str(resamples),
            "sha256": sha256(resamples),
            "seed": int(r["seed"]),
            "n_resamples": len(weights),
            "stratified": "positive and negative slides resampled separately with replacement",
            "slides_and_strata_match": True,
        },
        "histogram": {
            "n_bins": n_bins,
            "logit_range": [lo, hi],
            "tolerance": float(config.histogram.tolerance),
            "max_abs_error": max_err,
            "errors": dict(zip(config.eval_runs, errors, strict=True)),
        },
    }
    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=str))
    (out / "report.md").write_text(report(results.iloc[0], runs, counts, checks, config))
    return out, checks


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------


def report(res: pd.Series, runs: pd.DataFrame, counts: dict, checks: dict, config: DictConfig) -> str:
    ci = f"{config.ci_level:.0%}"
    lines = [
        f"# tab:res-nucleus: {config.dataset_label} test",
        "",
        "Model: shape embedding (efd), rotated values (`vrope`), k-NN attention, k = 24; three seeds.",
        "",
        "## Runs (step 0)",
        "",
        "| Seed | Training run | Evaluation run | Config (embedding, values, attention, k) | Checkpoint (best validation loss) | Training |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in runs.itertuples():
        values = "rotated values" if r.attention_variant == "vrope" and r.use_rope == "True" else r.attention_variant
        status = r.training_status if r.training_status == "FINISHED" else f"{r.training_status} (did not finish)"
        training = f"{status}, last epoch {r.last_logged_epoch} of {r.max_epochs}"
        if isinstance(r.training_resumed_from, str) and r.training_resumed_from:
            training += f", resumed from {r.training_resumed_from[:8]}"
        lines.append(
            f"| {r.seed} | `{r.training_run}` | `{r.eval_run}` | {r.embedding_mode}, {values}, "
            f"{r.context}, {r.k} | epoch {r.checkpoint_epoch} | {training} |"
        )
    trained_before = runs["trained_before_block_mask_fix"].all()
    lines += [
        "",
        f"Checkpoints are selected on `{config.checkpoint_monitor}` of the validation split of the "
        "training metadata only; each evaluation loads its training run's single checkpoint, at its best "
        "validation epoch.",
        f"All three were trained before the block-mask fix ({config.block_mask_fix_date})"
        if trained_before
        else "Block-mask fix: see runs.csv.",
        "and evaluated after it." if trained_before and not runs["evaluated_before_block_mask_fix"].any() else "",
        "",
        "## Labels",
        "",
        "| Test slides | Positive / negative slides | Labelled nuclei | Positive nuclei | Ignored nuclei | Prevalence |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| {counts['test_slides']} | {counts['positive_slides']} / {counts['negative_slides']} | "
        f"{counts['labeled_nuclei']:,} | {counts['positive_nuclei']:,} | {counts['ignored_nuclei']} | "
        f"{counts['prevalence']:.4f} |",
        "",
        "## Results",
        "",
        f"| Metric | Seed mean [{ci} CI] | Seeds | SD over seeds |",
        "| --- | --- | --- | --- |",
    ]
    for m in METRICS:
        seeds = ", ".join(f"{res[f'{m}_seed_{k}']:.4f}" for k in (1, 2, 3))
        lines.append(
            f"| {m} | {res[m]:.4f} [{res[f'{m}_ci_lo']:.4f}, {res[f'{m}_ci_hi']:.4f}] | {seeds} | "
            f"{res[f'{m}_seed_sd']:.4f} |"
        )
    h, rs = checks["histogram"], checks["resamples"]
    lines += [
        "",
        f"CIs: percentile, B = {rs['n_resamples']:,} slide resamples stratified by positive/negative slide "
        f"(canonical file `{rs['file']}`, seed {rs['seed']}, sha256 {rs['sha256'][:16]}…), seed-averaged metric "
        "per resample.",
        f"Histogram check: {h['n_bins']:,} bins over the logit range; max |histogram − exact torchmetrics| = "
        f"{h['max_abs_error']:.1e} (tolerance {h['tolerance']:.0e}).",
        "",
    ]
    return "\n".join(lines)


def preflight(out: Path, slide_ids: list[str]) -> None:
    """No credentials in any text output; no slide ids outside the resamples file."""
    ids = set(slide_ids)
    for path in out.iterdir():
        if path.suffix not in {".csv", ".json", ".md", ".yaml"}:
            continue
        text = path.read_text()
        assert not CREDENTIAL_URL.search(text), f"credential URL in {path.name}"
        assert not any(i in text for i in ids), f"slide id in {path.name}"


def code_commit() -> str:
    repo = Path(__file__).parents[1]
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    dirty = git("status", "--porcelain", "--untracked-files=all", "--", *CODE)
    return f"{git('rev-parse', 'HEAD')}{' (dirty)' if dirty else ''}"


@with_cli_args(["+postprocessing=nucleus_results"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    client = logger.experiment
    client.set_tag(logger.run_id, "code_commit", code_commit())
    out, checks = compute(config, client)

    runs = pd.read_csv(out / "runs.csv")
    params = {"resamples_sha256": checks["resamples"]["sha256"]}
    for r in runs.itertuples():
        params |= {f"eval_run_{r.seed}": r.eval_run, f"training_run_{r.seed}": r.training_run}
    logger.log_hyperparams(params)
    res = pd.read_csv(out / "nucleus_results.csv").iloc[0]
    logger.log_metrics(
        {f"test/{k}": float(res[k]) for k in res.index if k not in {"dataset", "patients"} and pd.notna(res[k])}
        | {"histogram/max_abs_error": checks["histogram"]["max_abs_error"]}
    )

    (out / "config-resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    slide_ids = list(np.load(Path(config.resamples))["slide_ids"])
    preflight(out, slide_ids)
    for name in ("nucleus_results.csv", "runs.csv", "checks.json", "report.md", "replicates.npz",
                 Path(config.resamples).name):
        logger.log_artifact(str(out / name))
    logger.log_artifact(str(out / "config-resolved.yaml"), "code")
    for source in CODE:
        logger.log_artifact(str(Path(__file__).parents[1] / source), "code")

    tags = client.get_run(logger.run_id).data.tags
    leaked = [k for k, v in tags.items() if CREDENTIAL_URL.search(v)]
    assert not leaked, f"credential URL in run tags {leaked}"


if __name__ == "__main__":
    main()
