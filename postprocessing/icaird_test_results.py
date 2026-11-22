# ruff: noqa: RUF001, RUF002  (Greek letters / minus signs in report text are intended)
r"""iCAIRD cervix test results for the thesis (NB-A-114).

Models (three seeds each, k-NN attention with k = 24, rotated values): Shape + RoVE (efd
embedding) and Position only + RoVE (blank token). Inference outputs only: the per-nucleus test
predictions stored in the evaluation runs, except shape seed 2, whose evaluation (7f555b52) used
epoch 567 while the run was still training; that checkpoint was later replaced by epoch 687 (lower
validation loss), so seed 2 was re-predicted at 687 on test and validation
(``postprocessing/nucleus_inference_at_commit.py``, model code unchanged since the evaluation
commits) and 687 is used everywhere (user decision, 2026-11-22). The epoch-567 metrics are reported
for reference.

0. Runs and test set. From the resolved configs: every evaluation predicts the same test table
   (mapping run 8ac7be17), with ``annotation_sparse`` on the re-annotation labels, k, context and
   values as expected; it loads its training run's checkpoint; the two models' training runs use
   the same train and validation slides. Every prediction set covers exactly the same nuclei
   (slide, 64-bit id prefix). The script stops otherwise.
1. Labels (the thesis rule for iCAIRD, as ``annotation_sparse``): on the 60 positive slides only
   label-1 nuclei (>= 75 % of the polygon inside the pathologist's regions) are scored, as 1; every
   other nucleus of these slides is ignored (the regions are not exhaustive). Every nucleus of the
   185 negative (normal / inflammation) slides is 0. Low grade is in no table.
2. Metrics per seed over the pooled labelled nuclei: AUPRC and AUROC (torchmetrics, exact) and the
   calibrated AUPRC at reference prevalence π₀ = 0.5 (Siblini et al. 2020; precision
   TP / (TP + c·FP), c = π(1 − π₀) / (π₀(1 − π)), π the prevalence of the evaluated set; as the
   PANDA Radboud and MMCI rows); seed mean and SD.
3. Bootstrap: slides resampled with replacement, positive and negative slides separately,
   B = 2,000, seed 20261108, generated once and shared by both models (no patient ids exist: one
   slide per case). Per resample each seed's metric weights each nucleus by its slide's
   multiplicity (calibrated AUPRC with the resample's prevalence), and the seeds are averaged;
   95 % percentile intervals. Paired Δ = shape − position only on the same resamples; two-sided
   p = 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0)). Replicates come from per-slide logit histograms on one uniform
   grid, refined until every seed's full-test-set metrics match the exact values.
4. Per-slide picture (shape): Youden threshold (max TPR − FPR over distinct scores, highest t on
   ties) of the seed-mean probability on the iCAIRD validation nuclei (training supervision, same
   sparse rule). Per positive slide: carcinoma share = label-1 nuclei / all nuclei of the slide
   (a lower bound: the regions are not exhaustive), recall at the threshold; FPR cannot be measured
   on positive slides (no labelled benign nuclei), so it is reported on the negative slides. The
   per-slide calibrated AUPRC (π₀ = pooled test prevalence) and AUROC rank the slide's carcinoma
   nuclei against all negative-slide nuclei (per seed, then the mean). Spearman ρ of recall with the
   share, CI from the positive stratum of the shared resamples.
5. Settings: the training configs vs the PANDA Radboud model.

Outputs (``out_dir``; logged to a new run in experiment 37): ``nucleus_results_icaird.csv``,
``ladder_icaird.csv``, ``per_slide_icaird.csv`` (anonymous slide index), ``share_bins_icaird.csv``,
``negative_slides_icaird.csv``, ``spearman_icaird.csv``, ``settings_icaird.csv``, ``runs.csv``,
``checks.json``, ``report.md``, ``tab_res_nucleus_icaird.tex``, ``replicates.npz`` and
``bootstrap_resamples_icaird_test.npz`` (multiplicities and strata only). No slide or patient id in
any output (checked before logging).
"""

import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import with_cli_args
from scipy.stats import rankdata

from postprocessing.bootstrap_sources import (
    bin_logits,
    bootstrap_p,
    hist_metrics,
    load_run,
    nucleus_keys,
    slide_histograms,
    stratified_resamples,
    to_logits,
)
from postprocessing.nucleus_results_mmci import CREDENTIAL_URL, exact_metrics
from postprocessing.patient_error_analysis_mmci import calibrated_ap
from postprocessing.patient_score_quantiles_mmci import youden_threshold
from postprocessing.tile.bootstrap_pixel_vs_nuclei import git_info


REPO = Path(__file__).parents[1]
CODE = (
    "postprocessing/icaird_test_results.py",
    "configs/postprocessing/icaird_test_results.yaml",
    "postprocessing/nucleus_inference_at_commit.py",  # seed-2 test and validation predictions
    "postprocessing/nucleus_inference_lowmem.py",  # the largest validation slide
)
METRICS = ("AUPRC", "AUPRC_cal", "AUROC")
CKPT = re.compile(r"mlflow-artifacts:/37/([0-9a-f]{32})/artifacts/checkpoints/epoch=(\d+)-step=\d+/")
SLIDE_ID = re.compile(r"IC-CX-\d+")


# --------------------------------------------------------------------------
# 0. Runs
# --------------------------------------------------------------------------


def resolved_config(client: MlflowClient, run_id: str, tmp: Path) -> dict:
    (tmp / run_id).mkdir(parents=True, exist_ok=True)
    path = client.download_artifacts(run_id, "configs/config-resolved.yaml", str(tmp / run_id))
    return yaml.safe_load(Path(path).read_text())


def callbacks(cfg: dict) -> list:
    cbs = cfg["trainer"].get("callbacks") or []
    return list(cbs.values()) if isinstance(cbs, dict) else list(cbs)


def check_runs(config: DictConfig, client: MlflowClient, tmp: Path) -> tuple[pd.DataFrame, dict]:
    ex = config.expected_eval
    rows, train_tables = [], {}
    for model, m in config.models.items():
        for seed, eval_id in enumerate(m.eval_runs, start=1):
            ev = client.get_run(eval_id)
            ecfg = resolved_config(client, eval_id, tmp)
            train_id = ev.data.tags["mlflow.parentRunId"]
            tr = client.get_run(train_id)
            tcfg = resolved_config(client, train_id, tmp)
            d, sup = ecfg["data"], ecfg["data"]["supervision"]["eval_strategy"]
            net = ecfg["model"]["net"]["config"]
            ck_run, ck_epoch = CKPT.search(ecfg["checkpoint"]).groups()
            monitors = [
                cb.get("monitor")
                for cb in callbacks(tcfg)
                if isinstance(cb, dict) and cb.get("_target_", "").endswith("ModelCheckpoint")
            ]
            stored = [f.path for f in client.list_artifacts(train_id, "checkpoints")]
            repl = (m.get("replacements") or {}).get(eval_id)
            row = {
                "model": model,
                "seed": seed,
                "eval_run": eval_id,
                "training_run": train_id,
                "predict_metadata": d["metadata"]["predict"],
                "supervision_mode": sup["mode"],
                "annot_dir": sup["annot_dir"],
                "embedding_mode": d["dataset"]["embedding_mode"],
                "train_embedding_mode": tcfg["data"]["dataset"]["embedding_mode"],
                "k": d["k"],
                "context": d["context"],
                "attention_variant": net.get("attention_variant"),
                "use_rope": net.get("use_rope"),
                "checkpoint_run": ck_run,
                "evaluated_epoch": int(ck_epoch),
                "checkpoint_monitor": ",".join(map(str, monitors)),
                "training_status": tr.info.status,
                "training_last_epoch": int(tr.data.metrics.get("epoch", -1)),
                "max_epochs": tcfg["trainer"]["max_epochs"],
                "best_validation_epoch_now": int(tr.data.metrics["best/epoch"]),
                "stored_checkpoints_now": ",".join(Path(s).name for s in stored),
                "train_table": tcfg["data"]["metadata"]["train"],
                "valid_table": tcfg["data"]["metadata"]["valid"],
                "replaced": repl is not None,
                "used_epoch": int(repl.checkpoint_epoch) if repl is not None else int(ck_epoch),
                "eval_commit": ev.data.tags.get("mlflow.source.git.commit"),
            }
            rows.append(row)
            for key in ("train", "valid"):
                train_tables.setdefault(key, set()).add(tcfg["data"]["metadata"][key])
    runs = pd.DataFrame(rows)

    for model, m in config.models.items():
        r = runs[runs["model"] == model]
        assert len(r) == 3 and r["training_run"].is_unique, f"{model}: need 3 distinct seeds"
        assert (r["embedding_mode"] == m.embedding_mode).all()
        assert (r["train_embedding_mode"] == m.embedding_mode).all()
    assert (runs["predict_metadata"] == config.metadata_uri).all(), "evaluations predict other test tables"
    assert (runs["supervision_mode"] == ex.supervision_mode).all()
    assert runs["annot_dir"].str.endswith(ex.annot_dir_suffix).all()
    assert runs["annot_dir"].nunique() == 1
    assert (runs["k"] == ex.k).all() and (runs["context"] == ex.context).all()
    assert (runs["attention_variant"] == ex.attention_variant).all()
    assert (runs["use_rope"] == ex.use_rope).all()
    assert (runs["checkpoint_run"] == runs["training_run"]).all(), "evaluation loads another run's checkpoint"
    assert (runs["checkpoint_monitor"] == config.checkpoint_monitor).all()
    # the stored checkpoint (single, best validation loss) is the evaluated one, or the replacement's
    for r in runs.itertuples():
        assert r.stored_checkpoints_now.startswith(f"epoch={r.used_epoch}-"), (
            f"{r.eval_run}: used epoch {r.used_epoch}, run holds {r.stored_checkpoints_now}"
        )
    # train / valid slides of both models' training runs
    tables = {}
    for key, uris in train_tables.items():
        ids = {u: frozenset(pd.read_parquet(download_artifacts(u))["slide_id"]) for u in uris}
        assert len(set(ids.values())) == 1, f"training runs use different {key} slides"
        tables[key] = {"uris": sorted(uris), "slides": len(next(iter(ids.values()))), "identical": True}
    return runs, tables


# --------------------------------------------------------------------------
# 1. Labels
# --------------------------------------------------------------------------


def sparse_labels(
    slides: pd.DataFrame, ref: pd.DataFrame, supervision_dir: Path, label_column: str
) -> tuple[NDArray[np.int8], dict]:
    """1 = label-1 nucleus of a positive slide, 0 = nucleus of a negative slide, −1 = ignored."""
    slide = ref["slide"].to_numpy()
    y = np.full(len(ref), -1, np.int8)
    neg = ~slides["is_carcinoma"].to_numpy(bool)
    y[neg[slide]] = 0
    bounds = np.searchsorted(slide, np.arange(len(slides) + 1))
    keys = ref["key"].to_numpy()
    label_rows_match = True
    for s, row in enumerate(slides.itertuples()):
        if not row.is_carcinoma:
            continue
        d = pd.read_parquet(supervision_dir / f"{row.slide_id}.parquet", columns=["id", label_column])
        assert set(d[label_column].unique()) <= {0, 1}
        k = nucleus_keys(d["id"])
        lo, hi = bounds[s], bounds[s + 1]
        pos = np.sort(k[d[label_column].to_numpy() == 1])
        idx = np.searchsorted(keys[lo:hi], pos)
        assert (idx < hi - lo).all() and (keys[lo:hi][idx] == pos).all(), (
            f"slide {s}: labelled nuclei without a prediction"
        )
        y[lo + idx] = 1
        label_rows_match &= len(d) == hi - lo
    info = {"label_rows_equal_predicted_nuclei_on_positive_slides": bool(label_rows_match)}
    return y, info


# --------------------------------------------------------------------------
# 2-3. Metrics, histograms, bootstrap
# --------------------------------------------------------------------------


def calibration(n_pos: NDArray | float, n_neg: NDArray | float, pi0: float) -> NDArray | float:
    pi = n_pos / (n_pos + n_neg)
    return pi * (1 - pi0) / (pi0 * (1 - pi))


def hist_calibrated(pos: NDArray, neg: NDArray, pi0: float) -> NDArray:
    """Calibrated AP from (..., bins) counts in ascending score order (steps as hist_metrics)."""
    c = np.asarray(calibration(pos.sum(-1), neg.sum(-1), pi0))
    tp, fp = np.cumsum(pos[..., ::-1], axis=-1), np.cumsum(neg[..., ::-1], axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + c[..., None] * fp), 0.0)
    return (pos[..., ::-1] * prec).sum(-1) / pos.sum(-1)


def hist_all(pos: NDArray, neg: NDArray, pi0: float) -> dict[str, NDArray]:
    out = hist_metrics(pos, neg)
    out["AUPRC_cal"] = hist_calibrated(pos, neg, pi0)
    return out


def replicates(pos: NDArray, neg: NDArray, weights: NDArray, pi0: float, chunk: int = 200) -> dict:
    out: dict[str, list] = {m: [] for m in METRICS}
    for start in range(0, len(weights), chunk):
        w = weights[start : start + chunk]
        for m, v in hist_all(w @ pos, w @ neg, pi0).items():
            out[m].append(v)
    return {m: np.concatenate(v) for m, v in out.items()}


def exact_all(y: NDArray, p: NDArray, pi0: float) -> dict[str, float]:
    e = exact_metrics(y, p)
    n_pos = int((y == 1).sum())
    e["AUPRC_cal"] = calibrated_ap(y, p, float(calibration(n_pos, len(y) - n_pos, pi0)))
    return e


def bootstrap_model(
    probs: NDArray, y: NDArray, slide: NDArray, n_slides: int, weights: NDArray, exact: list[dict],
    config: DictConfig, lo: float, hi: float,
) -> tuple[dict[str, NDArray], dict]:
    """(seeds, resamples) replicates of each metric; histogram grid refined until exact."""
    pi0 = float(config.pi0_reference)
    n_bins = int(config.histogram.n_bins)
    while True:
        hists, errors = [], []
        for p, e in zip(probs, exact, strict=True):
            pos, neg = slide_histograms(slide, bin_logits(to_logits(p), lo, hi, n_bins), y, n_slides, n_bins)
            full = hist_all(pos.sum(0), neg.sum(0), pi0)
            errors.append({m: float(full[m] - e[m]) for m in METRICS})
            hists.append((pos, neg))
        max_err = max(abs(v) for err in errors for v in err.values())
        print(f"grid {n_bins} bins over logit [{lo:.3f}, {hi:.3f}]: max |error| {max_err:.2e}", flush=True)
        if max_err <= config.histogram.tolerance or n_bins >= config.histogram.max_bins:
            break
        del hists
        n_bins *= 2
    assert max_err <= config.histogram.tolerance, "histogram grid too coarse at max_bins"
    rep = {m: np.empty((len(probs), len(weights))) for m in METRICS}
    for i, (pos, neg) in enumerate(hists):
        for m, v in replicates(pos, neg, weights, pi0).items():
            rep[m][i] = v
    assert all(np.isfinite(rep[m]).all() for m in METRICS)
    return rep, {"n_bins": n_bins, "logit_range": [lo, hi], "max_abs_error": max_err, "errors": errors}


# --------------------------------------------------------------------------
# 4. Per slide
# --------------------------------------------------------------------------


def spearman(x: NDArray, v: NDArray) -> float:
    return float(np.corrcoef(rankdata(x), rankdata(v))[0, 1])


def per_slide(
    config: DictConfig, slides: pd.DataFrame, y_all: NDArray, slide_all: NDArray, probs_all: NDArray,
    t: float, pi0_pooled: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Positive slides (share, recall, slide-vs-pooled-negatives metrics) and negative slides (FPR)."""
    n_slides = len(slides)
    total = np.bincount(slide_all, minlength=n_slides)
    mean = probs_all.astype(np.float64).mean(0)
    labelled = y_all >= 0
    neg_sel = y_all == 0
    neg_probs = probs_all[:, neg_sel]
    neg_mean = mean[neg_sel]
    bounds = np.searchsorted(slide_all, np.arange(n_slides + 1))
    pos_rows, neg_rows = [], []
    for s, row in enumerate(slides.itertuples()):
        sel = slice(bounds[s], bounds[s + 1])
        ys, ms = y_all[sel], mean[sel]
        if row.is_carcinoma:
            is1 = ys == 1
            n1 = int(is1.sum())
            seeds = []
            for k in range(len(probs_all)):
                p = np.concatenate([probs_all[k, sel][is1], neg_probs[k]])
                yy = np.r_[np.ones(n1, np.int8), np.zeros(neg_probs.shape[1], np.int8)]
                c = float(calibration(n1, neg_probs.shape[1], pi0_pooled))
                seeds.append(
                    {
                        "AUPRC_cal_vs_negatives": calibrated_ap(yy, p, c),
                        "AUROC_vs_negatives": exact_metrics(yy, p)["AUROC"],
                    }
                )
            pos_rows.append(
                {
                    "_stratum": len(pos_rows),  # position among the positive slides (dropped)
                    "category": row.category,
                    "nuclei": int(total[s]),
                    "carcinoma_nuclei": n1,
                    "ignored_nuclei": int((ys == -1).sum()),
                    "carcinoma_share": n1 / total[s],
                    "recall_youden": float((ms[is1] >= t).mean()),
                    **pd.DataFrame(seeds).mean().to_dict(),
                    "metrics_reliable": n1 >= int(config.min_carcinoma_nuclei),
                }
            )
        else:
            assert (ys == 0).all()
            neg_rows.append(
                {"nuclei": int(total[s]), "FPR_youden": float((ms >= t).mean())}
            )
    pos = pd.DataFrame(pos_rows)
    bins = np.asarray(config.share_bins, float)
    idx = np.clip(np.searchsorted(bins, pos["carcinoma_share"], side="right") - 1, 0, len(bins) - 2)
    pos["share_bin"] = np.asarray(config.share_bin_labels)[idx]
    neg = pd.DataFrame(neg_rows)
    # pooled FPR of the negative slides at t (nucleus level)
    assert abs(float((neg_mean >= t).mean()) - np.average(neg["FPR_youden"], weights=neg["nuclei"])) < 1e-9
    assert int(labelled.sum()) == int(pos["carcinoma_nuclei"].sum() + neg["nuclei"].sum())
    pos = pos.sort_values("carcinoma_share", ascending=False, kind="stable", ignore_index=True)
    pos.insert(0, "slide", np.arange(1, len(pos) + 1))  # anonymous index, by carcinoma share
    neg = neg.sort_values("FPR_youden", ascending=False, ignore_index=True)
    neg.insert(0, "slide", np.arange(1, len(neg) + 1))
    return pos, neg


def bin_table(config: DictConfig, pos: pd.DataFrame, neg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for b in reversed(list(config.share_bin_labels)):
        d = pos[pos["share_bin"] == b]
        r = d[d["metrics_reliable"]]
        rows.append(
            {
                "carcinoma_share": b,
                "slides": len(d),
                "median_recall": float(d["recall_youden"].median()) if len(d) else np.nan,
                "median_FPR": np.nan,  # no labelled benign nuclei on positive slides
                "median_AUPRC_cal": float(r["AUPRC_cal_vs_negatives"].median()) if len(r) else np.nan,
                "median_AUROC": float(r["AUROC_vs_negatives"].median()) if len(r) else np.nan,
                "slides_with_metrics": len(r),
            }
        )
    rows.append(
        {
            "carcinoma_share": "none",
            "slides": len(neg),
            "median_recall": np.nan,
            "median_FPR": float(neg["FPR_youden"].median()),
            "median_AUPRC_cal": np.nan,
            "median_AUROC": np.nan,
            "slides_with_metrics": 0,
        }
    )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 5. Settings
# --------------------------------------------------------------------------


def flat(d: object, prefix: str = "") -> dict:
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out |= flat(v, f"{prefix}.{k}" if prefix else str(k))
    else:
        out[prefix] = d
    return out


SETTINGS_SKIP = re.compile(
    r"^(logger|metadata|dataset|mlflow|data_path|project_path|seed|checkpoint|mode)\b|"
    r"\.(metadata|paths|annot_dir|cam_dir|pred_dir|ignore_dir)\b|path|num_workers"
)


def settings(client: MlflowClient, runs: pd.DataFrame, radboud_run: str, tmp: Path) -> pd.DataFrame:
    ref = flat(resolved_config(client, radboud_run, tmp))
    rows = []
    for model in runs["model"].unique():
        t = runs[runs["model"] == model]["training_run"].iloc[0]
        cfg = flat(resolved_config(client, t, tmp))
        for k in sorted(set(ref) | set(cfg)):
            if SETTINGS_SKIP.search(k):
                continue
            if ref.get(k) != cfg.get(k):
                rows.append({"model": model, "setting": k, "radboud": ref.get(k), "icaird": cfg.get(k)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Main computation
# --------------------------------------------------------------------------


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def load_predictions(config: DictConfig, slide_ids: list[str]) -> dict[str, list[pd.DataFrame]]:
    out = {}
    for model, m in config.models.items():
        frames = []
        for eval_id in m.eval_runs:
            repl = (m.get("replacements") or {}).get(eval_id)
            if repl is not None:
                frames.append(pd.read_parquet(repl.test_predictions))
            else:
                df, info = load_run(eval_id, slide_ids, Path(config.cache_dir), config.predictions_artifact)
                assert info["test_slides_without_predictions"] == 0
                assert info["files_outside_test_metadata"] == 0
                frames.append(df)
        out[model] = frames
    return out


def valid_predictions(config: DictConfig, vslides: pd.DataFrame) -> list[pd.DataFrame]:
    """Each seed's validation predictions, parts merged; slide = row of `vslides`."""
    d = Path(config.inference_dir)
    index = {sid: i for i, sid in enumerate(vslides["slide_id"])}
    shards = {k: pd.read_parquet(d / f"valid_{k}_slides.parquet") for k in config.valid_parts}
    ids = [sid for sh in shards.values() for sid in sh["slide_id"]]
    assert sorted(ids) == sorted(index), "validation parts != validation slides"
    out = []
    for run in config.valid_training_runs:
        parts = []
        for k, sh in shards.items():
            f = pd.read_parquet(d / f"{run}_valid_{k}.parquet")
            f["slide"] = sh["slide_id"].map(index).to_numpy()[f["slide"].to_numpy()].astype(np.int16)
            parts.append(f[["slide", "key", "p"]])
        out.append(pd.concat(parts, ignore_index=True).sort_values(["slide", "key"], ignore_index=True))
    return out


def compute(config: DictConfig, client: MlflowClient) -> tuple[Path, dict, list[str]]:
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp"
    runs, tables = check_runs(config, client, tmp)
    settings_df = settings(client, runs, config.radboud_training_run, tmp)
    shutil.rmtree(tmp, ignore_errors=True)

    slides = pd.read_parquet(download_artifacts(config.metadata_uri)).sort_values("slide_id", ignore_index=True)
    slide_ids = slides["slide_id"].tolist()
    n_slides = len(slides)

    preds = load_predictions(config, slide_ids)
    ref = preds["shape"][0][["slide", "key"]]
    for frames in preds.values():
        for f in frames:
            assert np.array_equal(f["slide"].to_numpy(), ref["slide"].to_numpy()), "different slides"
            assert np.array_equal(f["key"].to_numpy(), ref["key"].to_numpy()), "different nuclei"
    nuclei_per_slide = np.bincount(ref["slide"].to_numpy(), minlength=n_slides)
    assert (nuclei_per_slide == slides["nuclei_count"].to_numpy()).all(), "predictions != index nuclei"

    y_all, label_info = sparse_labels(slides, ref, Path(config.test_supervision_dir), config.label_column)
    slide_all = ref["slide"].to_numpy()
    probs_all = {m: np.stack([f["p"].to_numpy() for f in fr]) for m, fr in preds.items()}
    del preds
    keep = y_all >= 0
    y, slide = y_all[keep], slide_all[keep]
    probs = {m: p[:, keep] for m, p in probs_all.items()}

    pos_per_slide = np.bincount(slide, weights=(y == 1), minlength=n_slides)
    lab_per_slide = np.bincount(slide, minlength=n_slides)
    positive = slides["is_carcinoma"].to_numpy(bool)
    assert np.array_equal(pos_per_slide > 0, positive), "a positive slide without label-1 nuclei"
    assert lab_per_slide.min() > 0, "a test slide without labelled nuclei"
    counts = {
        "test_slides": n_slides,
        "positive_slides": int(positive.sum()),
        "negative_slides": int((~positive).sum()),
        "patients": None,  # one slide per case; no patient ids
        "test_nuclei": len(y_all),
        "labeled_nuclei": len(y),
        "positive_nuclei": int((y == 1).sum()),
        "negative_nuclei": int((y == 0).sum()),
        "ignored_nuclei": int((~keep).sum()),
        "prevalence": float((y == 1).mean()),
        "positive_slide_label1_min": int(pos_per_slide[positive].min()),
        "positive_slide_label1_median": float(np.median(pos_per_slide[positive])),
        **label_info,
    }
    summary = pd.read_csv(Path(config.test_supervision_dir) / "_summary.csv")
    assert set(summary["slide_id"]) == set(slides.loc[positive, "slide_id"])
    assert int(summary["inside"].sum()) == counts["positive_nuclei"], "label-1 nuclei != label builder summary"
    counts["label1_matches_builder_summary"] = True
    print(json.dumps(counts, indent=1), flush=True)

    # exact metrics -------------------------------------------------------------
    pi0 = float(config.pi0_reference)
    exact = {m: [exact_all(y, p, pi0) for p in probs[m]] for m in probs}
    # the replaced evaluation (epoch 567), for reference
    replaced = {}
    for m in config.models.values():
        for eval_id in m.get("replacements") or {}:
            df, _ = load_run(eval_id, slide_ids, Path(config.cache_dir), config.predictions_artifact)
            assert np.array_equal(df["key"].to_numpy(), ref["key"].to_numpy())
            replaced[eval_id] = exact_all(y, df["p"].to_numpy()[keep], pi0)
    print("exact:", json.dumps(exact, indent=1), "replaced:", replaced, flush=True)

    # shared resamples ------------------------------------------------------------
    b = config.bootstrap
    mult = stratified_resamples(positive, int(b.n_resamples), int(b.seed))
    weights = mult.astype(np.float64)
    np.savez_compressed(
        out / "bootstrap_resamples_icaird_test.npz",
        multiplicities=mult, positive=positive, seed=int(b.seed),
        note=np.array("columns = test slides sorted by slide id (ids not stored)"),
    )
    finite = np.concatenate([lg[np.isfinite(lg)] for p in probs.values() for lg in map(to_logits, p)])
    lo, hi = float(finite.min()), float(finite.max())
    del finite
    rep, grids = {}, {}
    for m, pm in probs.items():
        rep[m], grids[m] = bootstrap_model(pm, y, slide, n_slides, weights, exact[m], config, lo, hi)
    np.savez_compressed(
        out / "replicates.npz", **{f"{m}/{k}": v for m in rep for k, v in rep[m].items()}
    )

    q = (1 - float(b.ci_level)) / 2 * 100
    results = []
    for m in probs:
        row = {"model": config.models[m].label, **{k: counts[k] for k in (
            "test_slides", "positive_slides", "negative_slides", "labeled_nuclei", "positive_nuclei",
            "ignored_nuclei", "prevalence")}}
        for met in METRICS:
            seeds = np.array([e[met] for e in exact[m]])
            ci = np.percentile(rep[m][met].mean(0), [q, 100 - q])
            row |= {met: float(seeds.mean()), f"{met}_ci_lo": float(ci[0]), f"{met}_ci_hi": float(ci[1]),
                    **{f"{met}_seed_{i + 1}": float(v) for i, v in enumerate(seeds)},
                    f"{met}_seed_sd": float(seeds.std(ddof=1))}
        results.append(row)
    results = pd.DataFrame(results)
    results.to_csv(out / "nucleus_results_icaird.csv", index=False)

    ladder = []
    for met in METRICS:
        d = rep["shape"][met].mean(0) - rep["position"][met].mean(0)
        point = float(np.mean([e[met] for e in exact["shape"]]) - np.mean([e[met] for e in exact["position"]]))
        ci = np.percentile(d, [q, 100 - q])
        ladder.append({"metric": met, "delta_shape_minus_position": point, "ci_lo": float(ci[0]),
                       "ci_hi": float(ci[1]), "p_bootstrap": bootstrap_p(d) if met == "AUPRC" else np.nan,
                       "share_resamples_delta_positive": float((d > 0).mean())})
    ladder = pd.DataFrame(ladder)
    ladder.to_csv(out / "ladder_icaird.csv", index=False)

    # 4. per slide ------------------------------------------------------------------
    vslides = pd.read_parquet(download_artifacts(config.valid_metadata_uri)).sort_values("slide_id", ignore_index=True)
    vframes = valid_predictions(config, vslides)
    vref = vframes[0][["slide", "key"]]
    for f in vframes[1:]:
        assert np.array_equal(f["key"].to_numpy(), vref["key"].to_numpy())
    assert (np.bincount(vref["slide"].to_numpy(), minlength=len(vslides)) == vslides["nuclei_count"].to_numpy()).all()
    vy, vinfo = sparse_labels(vslides, vref, Path(config.valid_supervision_dir), config.label_column)
    vmean = np.stack([f["p"].to_numpy() for f in vframes]).astype(np.float64).mean(0)
    vkeep = vy >= 0
    t, j = youden_threshold(vy[vkeep], vmean[vkeep])
    vpred = vmean[vkeep] >= t
    threshold = {
        "youden_threshold": t,
        "youden_J": j,
        "validation_slides": len(vslides),
        "validation_positive_slides": int(vslides["is_carcinoma"].sum()),
        "validation_labeled_nuclei": int(vkeep.sum()),
        "validation_positive_nuclei": int((vy == 1).sum()),
        "validation_recall": float(vpred[vy[vkeep] == 1].mean()),
        "validation_FPR": float(vpred[vy[vkeep] == 0].mean()),
        **{f"validation_{k}": v for k, v in vinfo.items()},
    }
    print(threshold, flush=True)
    del vframes, vmean, vy

    pos, neg = per_slide(config, slides, y_all, slide_all, probs_all["shape"], t, counts["prevalence"])
    # Spearman CIs from the positive stratum of the shared resamples, columns in `pos` order
    m_pos = mult[:, np.flatnonzero(positive)][:, pos.pop("_stratum").to_numpy()]
    bins = bin_table(config, pos, neg)
    pos.to_csv(out / "per_slide_icaird.csv", index=False)
    neg.to_csv(out / "negative_slides_icaird.csv", index=False)
    bins.to_csv(out / "share_bins_icaird.csv", index=False)

    corr = []
    for col in ("recall_youden", "AUPRC_cal_vs_negatives", "AUROC_vs_negatives"):
        x, v = pos["carcinoma_share"].to_numpy(), pos[col].to_numpy()
        reps = np.array([spearman(np.repeat(x, w), np.repeat(v, w)) for w in m_pos])
        reps = reps[np.isfinite(reps)]
        ci = np.percentile(reps, [q, 100 - q])
        corr.append({"variable": col, "n_slides": len(x), "spearman_rho": spearman(x, v),
                     "ci_lo": float(ci[0]), "ci_hi": float(ci[1]), "valid_resamples": len(reps)})
    corr.append({"variable": "FPR_youden", "n_slides": 0, "spearman_rho": np.nan, "ci_lo": np.nan,
                 "ci_hi": np.nan, "valid_resamples": 0})
    corr = pd.DataFrame(corr)
    corr.to_csv(out / "spearman_icaird.csv", index=False)

    neg_summary = {
        "slides": len(neg),
        "pooled_FPR": float(np.average(neg["FPR_youden"], weights=neg["nuclei"])),
        "median_FPR": float(neg["FPR_youden"].median()),
        "q1_FPR": float(neg["FPR_youden"].quantile(0.25)),
        "q3_FPR": float(neg["FPR_youden"].quantile(0.75)),
        "max_FPR": float(neg["FPR_youden"].max()),
        "slides_FPR_above_0.05": int((neg["FPR_youden"] > 0.05).sum()),
        "pooled_recall_positive_slides": float(
            np.average(pos["recall_youden"], weights=pos["carcinoma_nuclei"])
        ),
    }

    runs.to_csv(out / "runs.csv", index=False)
    settings_df.to_csv(out / "settings_icaird.csv", index=False)
    seed2 = Path(next(iter(config.models.shape.replacements.values())).test_predictions)
    checks = {
        "counts": counts,
        "tables": tables,
        "same_test_slides_and_nuclei": True,
        "labels": "positive slides: annot_label = 1 of the re-annotation labels (>= 75 % of the polygon inside "
        "the pathologist's regions) scored as 1, every other nucleus ignored; negative slides: all 0",
        "replaced_evaluations": {
            k: {"exact_epoch_567": v, "reason": "checkpoint replaced while training; re-predicted at the stored epoch",
                "replacement_sha256": sha256(seed2)}
            for k, v in replaced.items()
        },
        "threshold": threshold,
        "negative_slides": neg_summary,
        "bootstrap": {"n_resamples": int(b.n_resamples), "seed": int(b.seed), "unit": "slide",
                      "stratified": "positive (60) and negative (185) slides separately",
                      "shared_by_models": True, "ci": f"{b.ci_level:.0%} percentile",
                      "p": "2 min(P(d <= 0), P(d >= 0))"},
        "histogram": {m: dict(g) for m, g in grids.items()},
        "pi0_reference": pi0,
    }
    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=str))
    (out / "report.md").write_text(report(results, ladder, pos, neg, bins, corr, runs, checks, settings_df, config))
    (out / "tab_res_nucleus_icaird.tex").write_text(latex_row(results.iloc[0]) + "\n")
    return out, checks, slide_ids


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------


def ci3(r: pd.Series, m: str) -> str:
    return f"{r[m]:.3f} [{r[f'{m}_ci_lo']:.3f}, {r[f'{m}_ci_hi']:.3f}]"


def latex_row(r: pd.Series) -> str:
    return (
        f"\\quad iCAIRD & {int(r.test_slides)} & {r.prevalence:.3f} & {r.AUPRC:.3f} "
        f"[{r.AUPRC_ci_lo:.3f}, {r.AUPRC_ci_hi:.3f}] & {r.AUPRC_cal:.3f} [{r.AUPRC_cal_ci_lo:.3f}, "
        f"{r.AUPRC_cal_ci_hi:.3f}] & {r.AUROC:.3f} [{r.AUROC_ci_lo:.3f}, {r.AUROC_ci_hi:.3f}] \\\\"
    )


def f3(v: float) -> str:
    return "—" if pd.isna(v) else f"{v:.3f}"


def report(results, ladder, pos, neg, bins, corr, runs, checks, settings_df, config) -> str:
    c, th, ns = checks["counts"], checks["threshold"], checks["negative_slides"]
    names = {"AUPRC": "AUPRC", "AUPRC_cal": "Calibrated AUPRC (π₀ = 0.5)", "AUROC": "AUROC"}
    L = [
        "# iCAIRD cervix test results (NB-A-114)",
        "",
        "## Test set",
        "",
        f"{c['test_slides']} slides ({c['positive_slides']} positive / {c['negative_slides']} negative); "
        f"{c['labeled_nuclei']:,} labelled nuclei ({c['positive_nuclei']:,} carcinoma, {c['negative_nuclei']:,} "
        f"benign); {c['ignored_nuclei']:,} unlabelled nuclei of positive slides ignored; prevalence "
        f"{c['prevalence']:.3f}. Both models predicted exactly the same slides and nuclei (asserted).",
        "",
        "## (1) tab:res-nucleus, Shape + RoVE",
        "",
        "| Metric | Seed mean [95% CI] | Seed 1 | Seed 2 | Seed 3 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for _, r in results.iterrows():
        if not r.model.startswith("Shape"):
            continue
        for m in METRICS:
            L.append(f"| {names[m]} | {ci3(r, m)} | " + " | ".join(f"{r[f'{m}_seed_{k}']:.3f}" for k in (1, 2, 3)) + " |")
    L += ["", "```latex", latex_row(results.iloc[0]), "```", ""]
    for k, v in checks["replaced_evaluations"].items():
        L.append(
            f"Seed 2 uses epoch 687 (re-predicted); the original evaluation `{k[:8]}` (epoch 567) gives AUPRC "
            f"{v['exact_epoch_567']['AUPRC']:.3f}, calibrated AUPRC {v['exact_epoch_567']['AUPRC_cal']:.3f}, "
            f"AUROC {v['exact_epoch_567']['AUROC']:.3f}."
        )
    L += [
        "",
        "## (2) Per-slide picture (Shape + RoVE)",
        "",
        f"Youden threshold on the validation nuclei (seed-mean probability; {th['validation_labeled_nuclei']:,} "
        f"labelled nuclei of {th['validation_slides']} slides, carcinoma prevalence "
        f"{th['validation_positive_nuclei'] / th['validation_labeled_nuclei']:.3f}): t = {th['youden_threshold']:.3f} "
        f"(J = {th['youden_J']:.3f}; validation recall {th['validation_recall']:.3f}, FPR "
        f"{th['validation_FPR']:.3f}). The validation labels are the coarse dataset annotations (whole regions), "
        "the test labels the pathologist's small precise regions.",
        "",
        "Positive slides have no labelled benign nuclei, so FPR is measured on the negative slides only; the "
        "per-slide calibrated AUPRC (π₀ = pooled test prevalence) and AUROC rank the slide's carcinoma nuclei "
        "against all negative-slide nuclei. Carcinoma share = label-1 nuclei / all nuclei of the slide (a lower "
        "bound: the regions are not exhaustive).",
        "",
        "| Carcinoma share | Slides | Median recall | Median FPR | Median cal. AUPRC | Median AUROC |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in bins.itertuples():
        n = f" ({r.slides_with_metrics})" if r.carcinoma_share != "none" else ""
        L.append(
            f"| {r.carcinoma_share} | {r.slides} | {f3(r.median_recall)} | {f3(r.median_FPR)} | "
            f"{f3(r.median_AUPRC_cal)}{n if not pd.isna(r.median_AUPRC_cal) else ''} | "
            f"{f3(r.median_AUROC)}{n if not pd.isna(r.median_AUROC) else ''} |"
        )
    L += [
        "",
        f"Negative slides: pooled FPR {ns['pooled_FPR']:.3f}, median {ns['median_FPR']:.3f} "
        f"[IQR {ns['q1_FPR']:.3f}, {ns['q3_FPR']:.3f}], max {ns['max_FPR']:.3f}; "
        f"{ns['slides_FPR_above_0.05']} of {ns['slides']} slides above 0.05. Pooled recall on positive slides "
        f"{ns['pooled_recall_positive_slides']:.3f}.",
        "",
        "| Spearman ρ with carcinoma share | ρ [95% CI] | Slides |",
        "| --- | --- | --- |",
    ]
    for r in corr.itertuples():
        txt = "not measurable (no labelled benign nuclei on positive slides)" if r.n_slides == 0 else (
            f"{r.spearman_rho:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]")
        L.append(f"| {r.variable} | {txt} | {r.n_slides} |")
    L += [
        "",
        "## (3) Information ladder (Q4): position only vs shape, same nuclei",
        "",
        "| Model | AUPRC [95% CI] | AUROC [95% CI] | Cal. AUPRC [95% CI] | AUPRC seeds | AUROC seeds |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in results.iterrows():
        L.append(
            f"| {r.model} | {ci3(r, 'AUPRC')} | {ci3(r, 'AUROC')} | {ci3(r, 'AUPRC_cal')} | "
            + ", ".join(f"{r[f'AUPRC_seed_{k}']:.3f}" for k in (1, 2, 3)) + " | "
            + ", ".join(f"{r[f'AUROC_seed_{k}']:.3f}" for k in (1, 2, 3)) + " |"
        )
    L += ["", "| Δ = shape − position only | Δ [95% CI] | p |", "| --- | --- | --- |"]
    for r in ladder.itertuples():
        p = "—" if pd.isna(r.p_bootstrap) else (f"<{2 / config.bootstrap.n_resamples:.3f}" if r.p_bootstrap == 0 else f"{r.p_bootstrap:.3f}")
        L.append(f"| {names[r.metric]} | {r.delta_shape_minus_position:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] | {p} |")
    L += [
        "",
        "## (4) Settings",
        "",
        "| Model | Seed | Training run | Evaluated epoch | Last logged epoch | max_epochs | Status |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in runs.itertuples():
        L.append(f"| {r.model} | {r.seed} | `{r.training_run[:8]}` | {r.used_epoch} | {r.training_last_epoch} | "
                 f"{r.max_epochs} | {r.training_status} |")
    L += ["", "Differences from the PANDA Radboud training config (seed c8f26853):", "",
          "| Model | Setting | Radboud | iCAIRD |", "| --- | --- | --- | --- |"]
    for r in settings_df.itertuples():
        L.append(f"| {r.model} | `{r.setting}` | {r.radboud} | {r.icaird} |")
    b = checks["bootstrap"]
    L += ["", f"Bootstrap: B = {b['n_resamples']}, seed {b['seed']}, slides stratified (positive / negative), "
          "shared by both models, seed mean inside each resample, percentile CIs; no patient ids (one slide per case).",
          ""]
    return "\n".join(L)


def preflight(out: Path, slide_ids: list[str]) -> None:
    ids = set(slide_ids)
    for path in out.iterdir():
        if path.is_dir():
            continue
        if path.suffix in {".csv", ".json", ".md", ".yaml", ".tex"}:
            text = path.read_text()
        elif path.suffix == ".npz":
            z = np.load(path)
            text = " ".join(map(str, z.files)) + " ".join(
                str(z[k]) for k in z.files if z[k].dtype.kind in "US"
            )
        else:
            continue
        assert not CREDENTIAL_URL.search(text), f"credential in {path.name}"
        assert not SLIDE_ID.search(text) and not any(i in text for i in ids), f"slide id in {path.name}"


ARTIFACTS = (
    "nucleus_results_icaird.csv", "ladder_icaird.csv", "per_slide_icaird.csv", "share_bins_icaird.csv",
    "negative_slides_icaird.csv", "spearman_icaird.csv", "settings_icaird.csv", "runs.csv", "checks.json",
    "report.md", "tab_res_nucleus_icaird.tex", "replicates.npz", "bootstrap_resamples_icaird_test.npz",
)


@with_cli_args(["+postprocessing=icaird_test_results"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
def main(config: DictConfig) -> None:
    os.environ["MLFLOW_TRACKING_URI"] = str(config.mlflow_tracking_uri)
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    client = MlflowClient()
    out, checks, slide_ids = compute(config, client)
    (out / "config-resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    preflight(out, slide_ids)
    if not config.log_mlflow:
        print(f"log_mlflow=false: outputs in {out}")
        return

    res = pd.read_csv(out / "nucleus_results_icaird.csv")
    ladder = pd.read_csv(out / "ladder_icaird.csv")
    runs = pd.read_csv(out / "runs.csv")
    params = {
        "dataset": config.dataset_label,
        "resamples_seed": config.bootstrap.seed,
        "n_resamples": config.bootstrap.n_resamples,
        "pi0_reference": config.pi0_reference,
        "youden_threshold": checks["threshold"]["youden_threshold"],
        "seed2_replacement": "7f555b52 (epoch 567) -> epoch 687, re-predicted",
        **{f"{r.model}_eval_run_{r.seed}": r.eval_run for r in runs.itertuples()},
        **{f"{r.model}_training_run_{r.seed}": r.training_run for r in runs.itertuples()},
        **{f"{r.model}_epoch_{r.seed}": r.used_epoch for r in runs.itertuples()},
    }
    tags = {"mlflow.runName": config.metadata.run_name, "mlflow.note.content": config.metadata.description,
            "notebook": "NB-A-114", **git_info(*CODE)}
    for text in [*map(str, tags.values()), *map(str, params.values())]:
        assert not CREDENTIAL_URL.search(text), "credential URL in a tag or param"
    run_id = client.create_run(str(config.metadata.experiment_id), run_name=config.metadata.run_name,
                               tags=tags).info.run_id
    user = client.get_run(config.models.shape.eval_runs[0]).data.tags.get("mlflow.user")
    if user:
        client.set_tag(run_id, "mlflow.user", user)
    for k, v in params.items():
        client.log_param(run_id, k, v)
    for r in res.itertuples():
        key = "shape" if r.model.startswith("Shape") else "position"
        for m in METRICS:
            for suffix in ("", "_ci_lo", "_ci_hi"):
                client.log_metric(run_id, f"test/{key}/{m}{suffix}", float(getattr(r, m + suffix)))
    for r in ladder.itertuples():
        client.log_metric(run_id, f"test/delta_shape_minus_position/{r.metric}", r.delta_shape_minus_position)
    client.log_metric(run_id, "test/prevalence", float(res["prevalence"].iloc[0]))
    for name in ARTIFACTS:
        client.log_artifact(run_id, str(out / name))
    client.log_artifact(run_id, str(out / "config-resolved.yaml"), "code")
    for source in CODE:
        client.log_artifact(run_id, str(REPO / source), "code")
    r = client.get_run(run_id)
    leaked = [k for k, v in {**r.data.tags, **r.data.params}.items() if CREDENTIAL_URL.search(v)]
    assert not leaked, f"credential URL in run tags/params {leaked}"
    client.set_terminated(run_id)
    print(f"MLflow run {run_id} (experiment {config.metadata.experiment_id}); outputs in {out}")


if __name__ == "__main__":
    main()
