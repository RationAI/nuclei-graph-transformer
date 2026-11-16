# ruff: noqa: RUF002  (Greek letters / minus signs in report text are intended)
r"""Slide-level error analysis on the MMCI TL test set: slide-supervised nuclei model vs attention-MIL.

Models (stored test predictions only):

- the slide-supervised nuclei model (Shape + RoVE + k-NN, k = 44, full slides, sparse supervision),
  three seeds: per-slide ``graph_prediction`` and per-nucleus ``nuclei_prediction`` of its evaluation
  runs (``eval_runs``); threshold of each seed = its validation F1 threshold (``threshold_metric``);
- the classic attention-MIL baselines (Virchow2, PGP; experiment 65): ``tables/sl_predictions.json``
  and the validation slide threshold of their config (``slide_threshold``).

Outputs (``out_dir``; one run in experiment 97, tag thesis=main). Slides are named by thesis patient
label and a running number within the patient (``P3 s2``), never by slide id.

1. ``slides_mmci.csv``: per slide the label, nuclei, ignored nuclei (Another pathology and Exclude),
   carcinoma nuclei and the share of carcinoma among the labelled nuclei, the scores of every model and
   seed (and the nuclei model's seed mean), and the errors at each threshold.
2. ``errors_mmci.csv`` (every FN and FP per model and seed) and ``error_counts_mmci.csv``.
3. ``share_analysis_mmci.csv``: Spearman ρ between a positive slide's carcinoma share and its score
   (nuclei model: seed mean), and the slide AUROC of the positives with share < ``small_share`` and
   ≥ ``small_share`` against all negatives (nuclei model: mean of the per-seed AUROCs).
4. ``intervals_mmci.csv`` and ``paired_difference_mmci.csv``: slide AUROC / AUPRC with percentile CIs over
   the patient resamples of run ``resamples_run`` (each slide weighted by its patient's multiplicity;
   the nuclei model as the mean of its seeds' metrics per resample), and nuclei model − MIL Virchow2.
5. ``nucleus_mmci.csv``: the nuclei model's nucleus AUROC / AUPRC on the test nuclei (Another pathology
   and Exclude ignored, as run e8682243), exact per seed and the seed mean, with CIs from the same
   resamples (per-slide logit histograms, checked against the exact metrics).
6. ``flagged_slides_mmci.csv``: the slides flagged in PC-A-104 (private list), by slide name.
"""

import hashlib
import json
import shutil
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import with_cli_args
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from postprocessing.bootstrap_sources import (
    bin_logits,
    hist_metrics,
    label_predictions,
    load_labels,
    nucleus_keys,
    replicate_metrics,
    slide_histograms,
    to_logits,
)
from postprocessing.nucleus_results_mmci import (
    CREDENTIAL_URL,
    exact_metrics,
    load_ignore,
    preflight,
)
from postprocessing.tile.bootstrap_pixel_vs_nuclei import git_info


REPO = Path(__file__).parents[1]
CODE = (
    "postprocessing/slide_error_analysis_mmci.py",
    "configs/postprocessing/slide_error_analysis_mmci.yaml",
)
METRICS = ("AUROC", "AUPRC")
NUCLEI = "nuclei model"


def slide_metrics(y: NDArray, s: NDArray, w: NDArray | None = None) -> dict[str, float]:
    return {
        "AUROC": float(roc_auc_score(y, s, sample_weight=w)),
        "AUPRC": float(average_precision_score(y, s, sample_weight=w)),
    }


def errors_at(y: NDArray, s: NDArray, t: float) -> dict:
    pred = s >= t
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "recall": tp / (tp + fn),
        "specificity": tn / (tn + fp),
    }


def value_of(df: pd.DataFrame, model: str, metric: str) -> float:
    row = df[(df["model"] == model) & (df["metric"] == metric)]
    assert len(row) == 1, (model, metric)
    return float(row["value"].iloc[0])


def ci(values: NDArray, level: float) -> tuple[float, float]:
    q = (1 - level) / 2 * 100
    lo, hi = np.percentile(values, [q, 100 - q])
    return float(lo), float(hi)


def read_eval_run(
    client: MlflowClient, run: str, slides: list[str], artifact: str, tmp: Path
) -> tuple[pd.Series, pd.DataFrame]:
    """(slide score per slide, (slide, key, p) of every nucleus sorted) of an evaluation run."""
    (tmp / run).mkdir(parents=True, exist_ok=True)
    local = Path(client.download_artifacts(run, artifact, str(tmp / run)))
    scores, frames = {}, []
    for s, slide_id in enumerate(slides):
        d = pq.read_table(
            local / f"{slide_id}.parquet",
            columns=["id", "nuclei_prediction", "graph_prediction"],
        ).to_pandas()
        g = d["graph_prediction"].unique()
        assert len(g) == 1, f"{run}: several slide scores in one slide"
        scores[slide_id] = float(g[0])
        assert d["id"].is_unique
        frames.append(
            pd.DataFrame(
                {
                    "slide": np.full(len(d), s, np.int16),
                    "key": nucleus_keys(d["id"]),
                    "p": d["nuclei_prediction"].to_numpy(np.float32),
                }
            )
        )
    shutil.rmtree(tmp / run, ignore_errors=True)
    nuclei = pd.concat(frames, ignore_index=True).sort_values(
        ["slide", "key"], ignore_index=True
    )
    return pd.Series(scores), nuclei


def compute(config: DictConfig, client: MlflowClient) -> tuple[Path, dict, list[str]]:
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp"
    tmp.mkdir(exist_ok=True)
    level = float(config.ci_level)

    meta = pd.read_parquet(download_artifacts(config.metadata_uri)).sort_values(
        "slide_id", ignore_index=True
    )
    meta["patient_id"] = meta["patient_id"].astype(str)
    slides = meta["slide_id"].tolist()
    y = meta["is_carcinoma"].to_numpy(bool)

    # patients and slide names -------------------------------------------------------------------
    mapping = pd.read_csv(config.patient_label_mapping, dtype=str)
    thesis = dict(config.thesis_from_old)
    assert dict(zip(mapping["patient"], mapping["thesis_label"], strict=True)) == thesis
    label_of = dict(zip(mapping["patient_id"], mapping["thesis_label"], strict=True))
    meta["patient"] = meta["patient_id"].map(label_of)
    assert meta["patient"].notna().all()
    meta["slide"] = (
        meta["patient"] + " s" + (meta.groupby("patient").cumcount() + 1).astype(str)
    )

    # carcinoma share (Another pathology and Exclude ignored) ------------------------------------
    sup, ign_dir = Path(config.supervision_dir), Path(config.ignore_dir)
    carc, ignored, ignored_pos = [], [], []
    for r in meta.itertuples():
        pos = (
            pd.read_parquet(
                sup / f"{r.slide_id}.parquet", columns=["id", config.label_column]
            )
            if r.is_carcinoma
            else pd.DataFrame(columns=["id", config.label_column])
        )
        pos_ids = set(pos.loc[pos[config.label_column] == 1, "id"])
        f = ign_dir / f"{r.slide_id}.parquet"
        ign_ids = (
            set(
                pd.read_parquet(f, columns=["id", "ignore_label"]).query(
                    "ignore_label == 1"
                )["id"]
            )
            if f.exists()
            else set()
        )
        carc.append(len(pos_ids - ign_ids))
        ignored.append(len(ign_ids))
        ignored_pos.append(len(pos_ids & ign_ids))
    meta["ignored_nuclei"] = ignored
    meta["labelled_nuclei"] = meta["nuclei_count"].astype(int) - meta["ignored_nuclei"]
    meta["carcinoma_nuclei"] = carc
    meta["carcinoma_share"] = meta["carcinoma_nuclei"] / meta["labelled_nuclei"]
    assert sum(ignored_pos) == 0, "an ignored nucleus is labelled 1"
    assert (meta.loc[meta["is_carcinoma"], "carcinoma_nuclei"] > 0).all()

    # scores and thresholds ---------------------------------------------------------------------
    models: dict[str, dict] = {}
    nuclei_frames = []
    for i, run in enumerate(config.eval_runs, start=1):
        scores, nuclei = read_eval_run(
            client, run, slides, config.predictions_artifact, tmp
        )
        nuclei_frames.append(nuclei)
        m = client.get_run(run).data.metrics
        models[f"{NUCLEI} seed {i}"] = {
            "run": run,
            "scores": scores.reindex(slides).to_numpy(),
            "threshold": float(m[config.threshold_metric]),
            "logged": {
                "AUROC": m["test_thresholded/graph_AUROC"],
                "AUPRC": m["test_thresholded/graph_AUPRC"],
                "recall": m["test_thresholded/graph_recall"],
                "specificity": m["test_thresholded/graph_specificity"],
            },
        }
    for name, run in config.mil_runs.items():
        j = json.loads(
            Path(
                client.download_artifacts(run, "tables/sl_predictions.json", str(tmp))
            ).read_text()
        )
        sl = (
            pd.DataFrame(j["data"], columns=j["columns"])
            if "columns" in j
            else pd.DataFrame(j)
        )
        sl = sl.set_index("slide")["sl_prediction"]
        assert set(sl.index) == set(slides), f"{name}: slides != test slides"
        cfg_path = client.download_artifacts(
            run, "configs/config-resolved.yaml", str(tmp)
        )
        import yaml

        thr = float(yaml.safe_load(Path(cfg_path).read_text())["slide_threshold"])
        m = client.get_run(run).data.metrics
        models[f"MIL {name}"] = {
            "run": run,
            "scores": sl.reindex(slides).to_numpy(np.float64),
            "threshold": thr,
            "logged": {
                "AUROC": m["sl_test/AUC"],
                "AUPRC": m["sl_test/AUPRC"],
                "recall": m["sl_test/recall"],
                "specificity": m["sl_test/specificity"],
            },
        }
    seeds = [k for k in models if k.startswith(NUCLEI)]
    seed_mean = np.mean([models[k]["scores"] for k in seeds], axis=0)

    # per-slide table, errors, counts -----------------------------------------------------------
    table = meta[
        [
            "slide",
            "patient",
            "is_carcinoma",
            "nuclei_count",
            "ignored_nuclei",
            "labelled_nuclei",
            "carcinoma_nuclei",
            "carcinoma_share",
        ]
    ].rename(columns={"is_carcinoma": "label_carcinoma", "nuclei_count": "nuclei"})
    error_rows, count_rows, checks_logged = [], [], {}
    for name, m in models.items():
        col = name.replace(" ", "_")
        table[f"score_{col}"] = m["scores"]
        pred = m["scores"] >= m["threshold"]
        err = np.where(y & ~pred, "FN", np.where(~y & pred, "FP", ""))
        table[f"error_{col}"] = err
        e = errors_at(y, m["scores"], m["threshold"])
        exact = slide_metrics(y, m["scores"])
        count_rows.append({"model": name, "threshold": m["threshold"], **e, **exact})
        for s in np.flatnonzero(err != ""):
            error_rows.append(
                {
                    "model": name,
                    "error": err[s],
                    "slide": table["slide"].iloc[s],
                    "patient": table["patient"].iloc[s],
                    "carcinoma_share": float(table["carcinoma_share"].iloc[s]),
                    "score": float(m["scores"][s]),
                    "threshold": m["threshold"],
                }
            )
        diffs = {
            "AUROC": abs(exact["AUROC"] - m["logged"]["AUROC"]),
            "AUPRC": abs(exact["AUPRC"] - m["logged"]["AUPRC"]),
            "recall": abs(e["recall"] - m["logged"]["recall"]),
            "specificity": abs(e["specificity"] - m["logged"]["specificity"]),
        }
        checks_logged[name] = diffs
        assert max(diffs.values()) < 1e-4, (name, diffs)
    table[f"score_{NUCLEI.replace(' ', '_')}_seed_mean"] = seed_mean
    counts = pd.DataFrame(count_rows)
    errors = pd.DataFrame(error_rows)

    # carcinoma share vs score -----------------------------------------------------------------
    small = float(config.small_share)
    pos_idx = np.flatnonzero(y)
    share = table["carcinoma_share"].to_numpy()
    groups = {
        f"positives share < {small:g}": pos_idx[share[pos_idx] < small],
        f"positives share >= {small:g}": pos_idx[share[pos_idx] >= small],
    }
    neg_idx = np.flatnonzero(~y)
    share_rows = []
    for name, scores in [
        (f"{NUCLEI} (seed mean)", seed_mean),
        ("MIL Virchow2", models["MIL Virchow2"]["scores"]),
        ("MIL PGP", models["MIL PGP"]["scores"]),
    ]:
        r = spearmanr(share[pos_idx], scores[pos_idx])
        row = {
            "model": name,
            "positive_slides": len(pos_idx),
            "spearman_rho_share_vs_score": float(r.statistic),
            "spearman_p": float(r.pvalue),
        }
        for g, idx in groups.items():
            sel = np.r_[idx, neg_idx]
            if name.startswith(NUCLEI):  # mean of the per-seed AUROCs
                auc = float(
                    np.mean(
                        [roc_auc_score(y[sel], models[k]["scores"][sel]) for k in seeds]
                    )
                )
            else:
                auc = float(roc_auc_score(y[sel], scores[sel]))
            row[f"slides ({g})"] = len(idx)
            row[f"AUROC ({g} vs all negatives)"] = auc
        share_rows.append(row)
    shares = pd.DataFrame(share_rows)

    # patient bootstrap of the slide metrics -----------------------------------------------------
    res_path = Path(
        download_artifacts(
            run_id=config.resamples_run, artifact_path=config.resamples_artifact
        )
    )
    sha = hashlib.sha256(res_path.read_bytes()).hexdigest()
    assert sha == config.resamples_sha256, "patient resamples changed"
    r = np.load(res_path)
    patients = [str(p) for p in r["patients"]]
    assert sorted(patients) == sorted(meta["patient_id"].unique())
    pidx = {p: i for i, p in enumerate(patients)}
    weights = r["multiplicities"][:, meta["patient_id"].map(pidx).to_numpy()].astype(
        np.float64
    )
    assert ((weights * y).sum(1) > 0).all() and ((weights * ~y).sum(1) > 0).all()
    rep = {k: {m: np.empty(len(weights)) for m in METRICS} for k in models}
    for b, w in enumerate(weights):
        keep = w > 0
        for k, m in models.items():
            v = slide_metrics(y[keep], m["scores"][keep], w[keep])
            for metric in METRICS:
                rep[k][metric][b] = v[metric]
    nuclei_rep = {m: np.mean([rep[k][m] for k in seeds], axis=0) for m in METRICS}
    int_rows = []
    for name in [NUCLEI, "MIL Virchow2", "MIL PGP"]:
        for metric in METRICS:
            if name == NUCLEI:
                per_seed = [
                    slide_metrics(y, models[k]["scores"])[metric] for k in seeds
                ]
                point, values = float(np.mean(per_seed)), nuclei_rep[metric]
            else:
                per_seed = []
                point, values = (
                    slide_metrics(y, models[name]["scores"])[metric],
                    rep[name][metric],
                )
            lo, hi = ci(values, level)
            int_rows.append(
                {
                    "model": name,
                    "metric": f"slide {metric}",
                    "value": point,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    **{f"seed_{i}": v for i, v in enumerate(per_seed, start=1)},
                    **(
                        {"seed_sd": float(np.std(per_seed, ddof=1))} if per_seed else {}
                    ),
                }
            )
    intervals = pd.DataFrame(int_rows)
    pd_rows = []
    for metric in METRICS:
        d = nuclei_rep[metric] - rep["MIL Virchow2"][metric]
        point = float(
            value_of(intervals, NUCLEI, f"slide {metric}")
            - value_of(intervals, "MIL Virchow2", f"slide {metric}")
        )
        lo, hi = ci(d, level)
        pd_rows.append(
            {
                "metric": f"slide {metric}",
                "difference (nuclei model - MIL Virchow2)": point,
                "ci_lo": lo,
                "ci_hi": hi,
                "share_of_resamples_le_0": float((d <= 0).mean()),
                "bootstrap_p_two_sided": float(
                    min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
                ),
            }
        )
    paired = pd.DataFrame(pd_rows)

    # nucleus AUROC of the nuclei model (Another pathology and Exclude ignored) ------------------
    keys = nuclei_frames[0][["slide", "key"]]
    for f in nuclei_frames[1:]:
        assert np.array_equal(f["slide"].to_numpy(), keys["slide"].to_numpy())
        assert np.array_equal(f["key"].to_numpy(), keys["key"].to_numpy())
    labels = load_labels(meta, sup, config.label_column)
    ny, unmatched = label_predictions(keys, labels, y)
    ign = load_ignore(meta, ign_dir)
    merged = keys.merge(
        ign[["slide", "key", "ignore"]], on=["slide", "key"], how="left", validate="1:1"
    )
    nign = merged["ignore"].eq(True).to_numpy()
    assert int(nign.sum()) == len(ign) and int(ny[nign].sum()) == 0
    keepn = ~nign
    nslide, ny = keys["slide"].to_numpy()[keepn], ny[keepn]
    probs = np.stack([f["p"].to_numpy()[keepn] for f in nuclei_frames])
    del nuclei_frames, merged, labels
    nexact = [exact_metrics(ny, p) for p in probs]
    logits = [to_logits(p) for p in probs]
    finite = np.concatenate([lg[np.isfinite(lg)] for lg in logits])
    lo_l, hi_l = float(finite.min()), float(finite.max())
    del finite
    n_bins = int(config.histogram.n_bins)
    while True:
        hists, errs = [], []
        for lg, e in zip(logits, nexact, strict=True):
            pos_h, neg_h = slide_histograms(
                nslide, bin_logits(lg, lo_l, hi_l, n_bins), ny, len(meta), n_bins
            )
            full = hist_metrics(pos_h.sum(0), neg_h.sum(0))
            errs.append(max(abs(float(full[m]) - e[m]) for m in METRICS))
            hists.append((pos_h, neg_h))
        if (
            max(errs) <= config.histogram.tolerance
            or n_bins >= config.histogram.max_bins
        ):
            break
        n_bins *= 2
    assert max(errs) <= config.histogram.tolerance, "histogram grid too coarse"
    nrep = {m: np.empty((len(probs), len(weights))) for m in METRICS}
    for i, (pos_h, neg_h) in enumerate(hists):
        for m, v in replicate_metrics(pos_h, neg_h, weights).items():
            nrep[m][i] = v
    del hists
    nuc_rows = []
    for m in METRICS:
        per = [e[m] for e in nexact]
        lo, hi = ci(nrep[m].mean(0), level)
        nuc_rows.append(
            {
                "model": f"{NUCLEI} (slide-supervised)",
                "metric": f"nucleus {m}",
                "value": float(np.mean(per)),
                "ci_lo": lo,
                "ci_hi": hi,
                **{f"seed_{i}": v for i, v in enumerate(per, start=1)},
                "seed_sd": float(np.std(per, ddof=1)),
            }
        )
    nucleus = pd.DataFrame(nuc_rows)

    # flagged slides (private list) -------------------------------------------------------------
    flagged_ids = pd.read_csv(config.flagged_slides)["slide_id"].tolist()
    assert set(flagged_ids) <= set(slides)
    flagged = table[meta["slide_id"].isin(flagged_ids).to_numpy()].copy()
    for name, m in models.items():
        flagged[f"threshold_{name.replace(' ', '_')}"] = m["threshold"]
    shutil.rmtree(tmp, ignore_errors=True)

    table.to_csv(out / "slides_mmci.csv", index=False)
    errors.to_csv(out / "errors_mmci.csv", index=False)
    counts.to_csv(out / "error_counts_mmci.csv", index=False)
    shares.to_csv(out / "share_analysis_mmci.csv", index=False)
    intervals.to_csv(out / "intervals_mmci.csv", index=False)
    paired.to_csv(out / "paired_difference_mmci.csv", index=False)
    nucleus.to_csv(out / "nucleus_mmci.csv", index=False)
    flagged.to_csv(out / "flagged_slides_mmci.csv", index=False)
    checks = {
        "eval_runs": list(config.eval_runs),
        "mil_runs": dict(config.mil_runs),
        "slides": len(meta),
        "positive_slides": int(y.sum()),
        "negative_slides": int((~y).sum()),
        "thresholds": {k: m["threshold"] for k, m in models.items()},
        "recomputed_minus_logged_abs": checks_logged,
        "carcinoma_share": "carcinoma nuclei / (slide nuclei - Another pathology and Exclude nuclei)",
        "resamples": {
            "run": config.resamples_run,
            "sha256": sha,
            "n_resamples": len(weights),
            "unit": "patient; each slide weighted by its patient's multiplicity",
        },
        "nucleus": {
            "ignored": int(nign.sum()),
            "labelled": len(ny),
            "positive": int(ny.sum()),
            "carcinoma_slide_nuclei_without_supervision_row": unmatched,
            "histogram_bins": n_bins,
            "histogram_max_abs_error": float(max(errs)),
        },
    }
    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=str))
    (out / "report.md").write_text(
        report(counts, errors, shares, intervals, paired, nucleus, checks)
    )
    private = [*mapping["patient_id"], *mapping["case_id"], *slides]
    return out, checks, private


def report(counts, errors, shares, intervals, paired, nucleus, checks) -> str:
    lines = [
        "# Slide-level error analysis: slide-supervised nuclei model vs attention-MIL (MMCI TL test)",
        "",
        (
            f"{checks['slides']} slides ({checks['positive_slides']} carcinoma, {checks['negative_slides']} benign); "
            "slides named by thesis patient label. Thresholds: each model's validation threshold."
        ),
        "",
        "## Errors",
        "",
        counts[
            [
                "model",
                "threshold",
                "FN",
                "FP",
                "recall",
                "specificity",
                "AUROC",
                "AUPRC",
            ]
        ]
        .round(4)
        .to_csv(index=False),
        "",
        errors.round(4).to_csv(index=False) if len(errors) else "No errors.",
        "",
        "## Carcinoma share vs score",
        "",
        shares.round(4).to_csv(index=False),
        "",
        "## Slide AUROC / AUPRC (95 % patient-bootstrap CIs)",
        "",
        intervals.round(4).to_csv(index=False),
        "",
        paired.round(4).to_csv(index=False),
        "",
        "## Nucleus AUROC / AUPRC of the slide-supervised nuclei model (Another pathology and Exclude ignored)",
        "",
        nucleus.round(4).to_csv(index=False),
        "",
    ]
    return "\n".join(lines)


@with_cli_args(["+postprocessing=slide_error_analysis_mmci"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
def main(config: DictConfig) -> None:
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    client = MlflowClient()
    out, checks, private = compute(config, client)
    (out / "config-resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    preflight(out, private)
    if not config.log_mlflow:
        print(f"log_mlflow=false: outputs in {out}")
        return
    params = {
        "dataset": config.dataset_label,
        **{f"eval_run_{i}": r for i, r in enumerate(config.eval_runs, start=1)},
        **{f"mil_run_{k}": v for k, v in config.mil_runs.items()},
        "threshold_metric": config.threshold_metric,
        "resamples_run": config.resamples_run,
        "resamples_sha256": checks["resamples"]["sha256"],
        "small_share": config.small_share,
        "ignore": "Another pathology and Exclude nuclei ignored (another_exclude files)",
    }
    tags = {
        "mlflow.runName": config.metadata.run_name,
        "mlflow.note.content": config.metadata.description,
        "thesis": "main",
        **git_info(*CODE),
    }
    for text in [*map(str, tags.values()), *map(str, params.values())]:
        assert not CREDENTIAL_URL.search(text), "credential URL in a tag or param"
    experiment = client.get_experiment_by_name(config.metadata.experiment_name)
    run_id = client.create_run(
        experiment.experiment_id, run_name=config.metadata.run_name, tags=tags
    ).info.run_id
    user = client.get_run(config.eval_runs[0]).data.tags.get("mlflow.user")
    if user:
        client.set_tag(run_id, "mlflow.user", user)
    for k, v in params.items():
        client.log_param(run_id, k, v)
    intervals = pd.read_csv(out / "intervals_mmci.csv")
    for r in intervals.itertuples():
        key = f"{r.model.replace(' ', '_')}/{r.metric.replace(' ', '_')}"
        client.log_metric(run_id, key, r.value)
        client.log_metric(run_id, f"{key}_ci_lo", r.ci_lo)
        client.log_metric(run_id, f"{key}_ci_hi", r.ci_hi)
    for name in (
        "slides_mmci.csv",
        "errors_mmci.csv",
        "error_counts_mmci.csv",
        "share_analysis_mmci.csv",
        "intervals_mmci.csv",
        "paired_difference_mmci.csv",
        "nucleus_mmci.csv",
        "flagged_slides_mmci.csv",
        "checks.json",
        "report.md",
    ):
        client.log_artifact(run_id, str(out / name))
    client.log_artifact(run_id, str(out / "config-resolved.yaml"), "code")
    for source in CODE:
        client.log_artifact(run_id, str(REPO / source), "code")
    r = client.get_run(run_id)
    leaked = [
        k
        for k, v in {**r.data.tags, **r.data.params}.items()
        if CREDENTIAL_URL.search(v)
    ]
    assert not leaked, f"credential URL in run tags/params {leaked}"
    client.set_terminated(run_id)
    print(
        f"MLflow run {run_id} (experiment {experiment.experiment_id}); outputs in {out}"
    )


if __name__ == "__main__":
    main()
