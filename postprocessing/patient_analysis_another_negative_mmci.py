# ruff: noqa: RUF002  (Greek letters / minus signs in report text are intended)
r"""Per-patient analysis on the MMCI TL test set with Another pathology nuclei negative at test.

Model: Shape + RoVE + k-NN attention (k = 44, agreement supervision), three seeds; training is
unchanged (Another pathology nuclei ignored by the training and validation supervision). Test labels
are those of run ``results_run`` (19c2ef40, tab:res-nucleus MMCI row): ``annot_label`` left-joined on
(slide, nucleus), missing -> 0, benign slides -> 0; the carcinoma annotation excludes Exclude and
Another pathology regions, so their nuclei are 0 and no nucleus is ignored. Nothing is predicted:
the stored per-nucleus test predictions of its evaluation runs.

Per patient (thesis labels P1-P10) and pooled:

1. Labelled and carcinoma nuclei, prevalence π; per seed, then the mean of the seeds: AUPRC and AUROC
   (torchmetrics, exact) and the calibrated AUPRC (Siblini et al., IDA 2020): average precision with
   precision TP / (TP + c·FP), c = π(1 − π₀) / (π₀(1 − π)), π₀ = the pooled test prevalence and 0.5.
   Check: c = 1 reproduces the torchmetrics AUPRC.
2. Recall, FPR, TP, FP, FN of the seed-mean test probability at the Youden threshold (and the
   recall-0.90 threshold) of runs ``quantiles_run`` / ``error_analysis_run``, chosen on the seed-mean
   validation scores (Another pathology ignored, as in training); the thresholds are taken as they
   are and checked against those runs' params.
3. Quantiles (``np.quantile``, linear) of the seed-mean test score per patient and class, in the format
   of ``patient_score_quantiles_mmci.csv``.
4. Hard negatives: on the test slides with Another pathology nuclei (``ignore_label = 1`` in the
   pipeline's ignore files), the FPR and median seed-mean score at the Youden threshold of those nuclei
   and of the other benign nuclei of the same slides.
5. Thesis order: whether P1-P10 still follow the descending share of carcinoma nuclei (prevalence).

Checks against ``results_run``: per-patient counts, prevalence, AUPRC and AUROC (old labels mapped to
the thesis labels), pooled per-seed AUPRC / AUROC, and the patient-bootstrap replicates (same patient
resamples, same logit histogram grid) and CIs.

Outputs (``out_dir``; logged to a new run in experiment 97): ``patient_analysis_mmci.csv``,
``patient_operating_points_mmci.csv``, ``patient_score_quantiles_mmci.csv``,
``hard_negatives_mmci.csv``, ``thesis_order_mmci.csv``, ``checks.json``, ``report.md``.
"""

import json
import shutil
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import with_cli_args

from postprocessing.bootstrap_sources import (
    bin_logits,
    label_predictions,
    load_labels,
    load_run,
    replicate_metrics,
    slide_histograms,
    to_logits,
)
from postprocessing.nucleus_results_mmci import (
    CREDENTIAL_URL,
    exact_metrics,
    preflight,
    sha256,
)
from postprocessing.patient_error_analysis_mmci import (
    calibrated_ap,
    calibration,
    ignored,
    operating_point,
)
from postprocessing.tile.bootstrap_pixel_vs_nuclei import git_info


REPO = Path(__file__).parents[1]
CODE = (
    "postprocessing/patient_analysis_another_negative_mmci.py",
    "configs/postprocessing/patient_analysis_another_negative_mmci.yaml",
)
METRICS = ("AUPRC", "AUROC")
OPS = ("recall", "FPR", "TP", "FP", "FN")


def reference(config: DictConfig, client: MlflowClient, tmp: Path) -> dict:
    """The results run (labels, eval runs, tables, replicates) and the threshold runs."""
    files = {
        name: client.download_artifacts(config.results_run, name, str(tmp))
        for name in (
            "checks.json",
            "runs.csv",
            "nucleus_results_mmci.csv",
            "nucleus_per_patient_mmci.csv",
            "replicates.npz",
        )
    }
    checks = json.loads(Path(files["checks.json"]).read_text())
    assert checks["counts"]["ignored_nuclei"] == 0, "results run must ignore no nucleus"
    assert checks["resamples"]["run"] == config.resamples_run
    runs = pd.read_csv(files["runs.csv"])
    eval_runs, training_runs = list(runs["eval_run"]), list(runs["training_run"])
    for e, t in zip(eval_runs, training_runs, strict=True):
        assert client.get_run(e).data.tags["mlflow.parentRunId"] == t

    th = {k: float(v) for k, v in config.thresholds.items()}
    q = client.get_run(config.quantiles_run).data.params
    a = client.get_run(config.error_analysis_run).data.params
    assert float(q["threshold_youden"]) == th["youden"]
    assert float(q["threshold_recall_0.90"]) == th["recall_0.90"]
    assert float(a["threshold"]) == th["recall_0.90"]
    assert q["parent_run"] == config.error_analysis_run
    a_eval = [a[f"eval_run_{i}"] for i in (1, 2, 3)]
    a_train = [a[f"training_run_{i}"] for i in (1, 2, 3)]
    assert a_eval == eval_runs and a_train == training_runs, (
        "threshold runs rest on other models"
    )
    rep = np.load(files["replicates.npz"])
    assert [str(r) for r in rep["eval_runs"]] == eval_runs
    return {
        "checks": checks,
        "eval_runs": eval_runs,
        "training_runs": training_runs,
        "pooled": pd.read_csv(files["nucleus_results_mmci.csv"]).iloc[0],
        "per_patient": pd.read_csv(files["nucleus_per_patient_mmci.csv"]),
        "replicates": {m: rep[m] for m in METRICS},
        "thresholds": th,
    }


def bootstrap(
    config: DictConfig,
    ref: dict,
    meta: pd.DataFrame,
    slide: NDArray,
    y: NDArray,
    probs: NDArray,
) -> dict:
    """Replicates over the results run's patient resamples, on its histogram grid; compared to its own."""
    res_path = Path(
        download_artifacts(
            run_id=config.resamples_run, artifact_path=config.resamples_artifact
        )
    )
    res_sha = sha256(res_path)
    assert res_sha == ref["checks"]["resamples"]["sha256"]
    r = np.load(res_path)
    assert int(r["seed"]) == int(config.resamples_seed)
    patients = [str(p) for p in r["patients"]]
    assert sorted(patients) == sorted(meta["patient_id"].unique())
    pidx = {p: i for i, p in enumerate(patients)}
    weights = r["multiplicities"][:, meta["patient_id"].map(pidx).to_numpy()].astype(
        np.float64
    )
    n_slides = len(meta)

    hist = ref["checks"]["histogram"]
    n_bins = int(hist["n_bins"])
    logits = [to_logits(p) for p in probs]
    finite = np.concatenate([lg[np.isfinite(lg)] for lg in logits])
    lo, hi = float(finite.min()), float(finite.max())
    del finite
    assert [lo, hi] == hist["logit_range"], "logit range != results run's"

    rep = {m: np.empty((len(probs), len(weights))) for m in METRICS}
    for i, lg in enumerate(logits):
        pos, neg = slide_histograms(
            slide, bin_logits(lg, lo, hi, n_bins), y, n_slides, n_bins
        )
        for m, v in replicate_metrics(pos, neg, weights).items():
            rep[m][i] = v
        del pos, neg
    q = (1 - float(config.ci_level)) / 2 * 100
    out = {
        "resamples_run": config.resamples_run,
        "resamples_sha256": res_sha,
        "n_resamples": len(weights),
        "unit": "patient (all slides and nuclei of a drawn patient, with multiplicity)",
        "histogram_bins": n_bins,
        "logit_range": [lo, hi],
    }
    for m in METRICS:
        diff = float(np.abs(rep[m] - ref["replicates"][m]).max())
        assert diff <= float(config.replicate_tolerance), (
            f"{m} replicates differ: {diff}"
        )
        ci = np.percentile(rep[m].mean(0), [q, 100 - q])
        out[m] = {
            "ci_lo": float(ci[0]),
            "ci_hi": float(ci[1]),
            "max_abs_diff_replicates_vs_results_run": diff,
            "ci_lo_results_run": float(ref["pooled"][f"{m}_ci_lo"]),
            "ci_hi_results_run": float(ref["pooled"][f"{m}_ci_hi"]),
        }
    return out


def compute(config: DictConfig, client: MlflowClient) -> tuple[Path, dict, list[str]]:
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp"
    tmp.mkdir(exist_ok=True)
    ref = reference(config, client, tmp)
    th = ref["thresholds"]

    # test: stored predictions and the results run's labels (no nucleus ignored)
    meta = pd.read_parquet(download_artifacts(config.metadata_uri)).sort_values(
        "slide_id", ignore_index=True
    )
    meta["patient_id"] = meta["patient_id"].astype(str)
    stored = [
        load_run(
            e,
            meta["slide_id"].tolist(),
            Path(config.test_cache_dir),
            config.predictions_artifact,
        )[0]
        for e in ref["eval_runs"]
    ]
    keys = stored[0][["slide", "key"]]
    for f in stored[1:]:
        assert np.array_equal(f["slide"].to_numpy(), keys["slide"].to_numpy())
        assert np.array_equal(f["key"].to_numpy(), keys["key"].to_numpy())
    carcinoma = meta["is_carcinoma"].to_numpy(bool)
    labels = load_labels(meta, Path(config.supervision_dir), config.label_column)
    y, unmatched = label_predictions(keys, labels, carcinoma)
    slide = keys["slide"].to_numpy()
    probs = np.stack([f["p"].to_numpy() for f in stored])
    another = ignored(meta, Path(config.another_pathology_dir), keys)
    del stored, labels
    shutil.rmtree(tmp, ignore_errors=True)

    counts = ref["checks"]["counts"]
    assert len(y) == counts["labeled_nuclei"]
    assert int(y.sum()) == counts["positive_nuclei"]
    assert unmatched == counts["carcinoma_slide_nuclei_without_supervision_row"]
    assert int(another.sum()) == ref["checks"]["regions"]["another_ignore_files"]
    assert int(y[another].sum()) == 0, "an Another pathology nucleus is labelled 1"

    # patients: old labels (results run) -> thesis labels
    mapping = pd.read_csv(config.patient_label_mapping, dtype=str)
    thesis = dict(config.thesis_from_old)
    assert dict(zip(mapping["patient"], mapping["thesis_label"], strict=True)) == thesis
    new_of = dict(zip(mapping["patient_id"], mapping["thesis_label"], strict=True))
    assert set(new_of) == set(meta["patient_id"])
    patient = meta["patient_id"].map(new_of).to_numpy()[slide]
    order = sorted(new_of.values(), key=lambda p: int(p[1:]))
    groups = [(p, patient == p) for p in order] + [("pooled", np.ones(len(y), bool))]

    pooled_pi = float(y.mean())
    pi0s = [pooled_pi if str(v) == "pooled" else float(v) for v in config.pi0]
    pi0_names = ["pooled" if str(v) == "pooled" else str(v) for v in config.pi0]
    tmean = probs.astype(np.float64).mean(0)
    tol = float(config.calibration_check_tolerance)

    # 1-2. per patient -----------------------------------------------------------------------
    rows, per_seed, ops_rows, max_check = [], [], [], 0.0
    for name, sel in groups:
        ys = y[sel]
        pi = float(ys.mean())
        seeds = []
        for p in probs:
            ps = p[sel]
            e = exact_metrics(ys, ps)
            max_check = max(max_check, abs(calibrated_ap(ys, ps, 1.0) - e["AUPRC"]))
            seeds.append(
                {
                    "AUPRC": e["AUPRC"],
                    **{
                        f"calibrated_AUPRC_pi0_{n}": calibrated_ap(
                            ys, ps, calibration(pi, q)
                        )
                        for n, q in zip(pi0_names, pi0s, strict=True)
                    },
                    "AUROC": e["AUROC"],
                }
            )
        per_seed.append({"patient": name, "seeds": seeds})
        yd = operating_point(ys, tmean[sel], th["youden"])
        rows.append(
            {
                "patient": name,
                "labelled_nuclei": int(sel.sum()),
                "carcinoma_nuclei": int(ys.sum()),
                "prevalence": pi,
                **pd.DataFrame(seeds).mean().to_dict(),
                **{f"{k}_youden": yd[k] for k in OPS},
            }
        )
        op_row: dict[str, object] = {"patient": name}
        for key in ("recall_0.90", "youden"):
            o = operating_point(ys, tmean[sel], th[key])
            suffix = "recall90" if key == "recall_0.90" else key
            op_row |= {f"{k}_{suffix}": o[k] for k in OPS}
        ops_rows.append(op_row)
    assert max_check <= tol, f"calibrated AP with c = 1 != AUPRC ({max_check:.2e})"
    table = pd.DataFrame(rows)
    ops = pd.DataFrame(ops_rows)

    # agreement with the results run (its per-patient table uses the old labels)
    pp = ref["per_patient"].copy()
    pp["patient"] = pp["patient"].map(thesis)
    pp = pp.set_index("patient").loc[order]
    t = table.set_index("patient")
    agree = {
        "labelled_nuclei_equal": bool(
            (pp["labeled_nuclei"] == t.loc[order, "labelled_nuclei"]).all()
        ),
        "carcinoma_nuclei_equal": bool(
            (pp["positive_nuclei"] == t.loc[order, "carcinoma_nuclei"]).all()
        ),
        "prevalence_max_abs_diff": float(
            (pp["prevalence"] - t.loc[order, "prevalence"]).abs().max()
        ),
        **{
            f"{m}_max_abs_diff": float((pp[m] - t.loc[order, m]).abs().max())
            for m in METRICS
        },
        **{
            f"pooled_{m}_seed_{k}_abs_diff": abs(
                float(ref["pooled"][f"{m}_seed_{k}"]) - per_seed[-1]["seeds"][k - 1][m]
            )
            for m in METRICS
            for k in (1, 2, 3)
        },
        "pooled_prevalence_abs_diff": abs(
            float(ref["pooled"]["prevalence"]) - pooled_pi
        ),
    }
    assert agree["labelled_nuclei_equal"] and agree["carcinoma_nuclei_equal"], agree
    assert agree["prevalence_max_abs_diff"] < 1e-12, agree
    assert agree["pooled_prevalence_abs_diff"] < 1e-12, agree
    # torchmetrics computes in float32: recomputations agree to a few float32 ulps (~1e-7)
    assert max(v for k, v in agree.items() if k.endswith("diff")) <= float(
        config.agreement_tolerance
    ), agree
    pooled = t.loc["pooled"]
    assert round(pooled["AUPRC"], 4) == 0.8956 and round(pooled["AUROC"], 4) == 0.9611

    # 3. quantiles -----------------------------------------------------------------------------
    qs = [float(q) for q in config.quantiles]
    qrows = []
    for p in order:
        sel = patient == p
        for cls, label in (("carcinoma", 1), ("benign", 0)):
            s = tmean[sel & (y == label)]
            qrows.append(
                {
                    "patient": p,
                    "class": cls,
                    "n": len(s),
                    "prevalence": float(y[sel].mean()),
                    **{
                        f"p{round(q * 100)}": float(v)
                        for q, v in zip(qs, np.quantile(s, qs), strict=True)
                    },
                }
            )
    quantiles = pd.DataFrame(qrows)
    assert quantiles["n"].sum() == len(y)

    # 4. hard negatives ------------------------------------------------------------------------
    ap_slides = np.unique(slide[another])
    hn_rows = []
    slide_label = {}
    for p in order:
        mine = [s for s in ap_slides if new_of[meta["patient_id"].iloc[s]] == p]
        for k, s in enumerate(mine, start=1):
            slide_label[s] = f"{p} slide {k}/{len(mine)}"

    def hard(name: str, sel_slides: NDArray) -> None:
        on = np.isin(slide, sel_slides)
        for group, mask in (
            ("another_pathology", on & another),
            ("other_benign", on & ~another & (y == 0)),
        ):
            s = tmean[mask]
            fp = int((s >= th["youden"]).sum())
            hn_rows.append(
                {
                    "slide": name,
                    "group": group,
                    "n": len(s),
                    "FP_youden": fp,
                    "FPR_youden": fp / len(s),
                    "median_score": float(np.median(s)),
                    "FPR_youden_seed_mean": float(
                        np.mean([(p[mask] >= th["youden"]).mean() for p in probs])
                    ),
                }
            )

    for s in sorted(ap_slides, key=lambda s: slide_label[s]):
        hard(slide_label[s], np.array([s]))
    hard("pooled", ap_slides)
    hard_neg = pd.DataFrame(hn_rows)

    # 5. thesis order --------------------------------------------------------------------------
    prev = t.loc[order, "prevalence"]
    rank = prev.rank(ascending=False, method="first").astype(int)
    order_tab = pd.DataFrame(
        {
            "patient": order,
            "prevalence": prev.to_numpy(),
            "rank_by_prevalence": rank.to_numpy(),
            "prevalence_another_ignored": [
                float(v) for v in thesis_prevalence_ignored(y, patient, another, order)
            ],
        }
    )
    order_tab["order_holds"] = order_tab["rank_by_prevalence"] == np.arange(
        1, len(order) + 1
    )

    # bootstrap (same patient resamples as the results run) ------------------------------------
    boot = bootstrap(config, ref, meta, slide, y, probs)

    table.to_csv(out / "patient_analysis_mmci.csv", index=False)
    ops.to_csv(out / "patient_operating_points_mmci.csv", index=False)
    quantiles.to_csv(out / "patient_score_quantiles_mmci.csv", index=False)
    hard_neg.to_csv(out / "hard_negatives_mmci.csv", index=False)
    order_tab.to_csv(out / "thesis_order_mmci.csv", index=False)
    checks = {
        "results_run": config.results_run,
        "error_analysis_run": config.error_analysis_run,
        "quantiles_run": config.quantiles_run,
        "eval_runs": ref["eval_runs"],
        "training_runs": ref["training_runs"],
        "labels": "annot_label left-joined on (slide_id, id), missing -> 0, is_carcinoma = False -> 0; "
        "Another pathology and Exclude nuclei labelled 0; no nucleus ignored (labels of the results run)",
        "test": {
            "slides": len(meta),
            "carcinoma_slides": int(carcinoma.sum()),
            "patients": len(order),
            "labelled_nuclei": len(y),
            "carcinoma_nuclei": int(y.sum()),
            "pooled_prevalence": pooled_pi,
            "another_pathology_nuclei": int(another.sum()),
            "another_pathology_slides": len(ap_slides),
        },
        "pi0": dict(zip(pi0_names, pi0s, strict=True)),
        "thresholds": {
            **th,
            "score": "seed-mean probability; chosen on the seed-mean validation score "
            "(Another pathology ignored, as the training supervision)",
        },
        "agreement_with_results_run": agree,
        "calibration_check": {"c": 1.0, "max_abs_diff_vs_torchmetrics": max_check},
        "bootstrap": boot,
        "thesis_order_holds": bool(order_tab["order_holds"].all()),
        "per_seed": per_seed,
    }
    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=str))
    (out / "report.md").write_text(report(table, hard_neg, order_tab, checks, config))
    private = [*mapping["patient_id"], *mapping["case_id"], *meta["slide_id"]]
    return out, checks, private


def thesis_prevalence_ignored(
    y: NDArray, patient: NDArray, another: NDArray, order: list[str]
) -> list[float]:
    """Prevalence per patient with the Another pathology nuclei removed (the thesis ranking's labels)."""
    keep = ~another
    return [float(y[keep & (patient == p)].mean()) for p in order]


def report(
    table: pd.DataFrame,
    hard_neg: pd.DataFrame,
    order_tab: pd.DataFrame,
    checks: dict,
    config: DictConfig,
) -> str:
    th, pi0, b = checks["thresholds"], checks["pi0"], checks["bootstrap"]
    pooled = table.set_index("patient").loc["pooled"]
    lines = [
        "# Per-patient analysis, Another pathology negative at test: MMCI TL test",
        "",
        (
            f"Labels of run `{config.results_run}` (Another pathology and Exclude nuclei labelled 0, none "
            "ignored); training unchanged. Shape + RoVE + k-NN attention (k = 44), three seeds, stored test "
            "predictions. AUPRC, calibrated AUPRC and AUROC per seed, then the mean; operating points on the "
            f"seed-mean probability at the Youden threshold {th['youden']} (run `{config.quantiles_run}`)."
        ),
        "",
        (
            f"Pooled prevalence (π₀) = {pi0['pooled']:.6f}; pooled AUPRC {pooled['AUPRC']:.4f} "
            f"[{b['AUPRC']['ci_lo']:.4f}, {b['AUPRC']['ci_hi']:.4f}], AUROC {pooled['AUROC']:.4f} "
            f"[{b['AUROC']['ci_lo']:.4f}, {b['AUROC']['ci_hi']:.4f}] (95 % patient-bootstrap CIs, "
            f"{b['n_resamples']:,} resamples of run `{config.resamples_run}`)."
        ),
        "",
        (
            "| Patient | Labelled | Carcinoma | Prevalence | AUPRC | Cal. AUPRC (π₀ pooled) | Cal. AUPRC (π₀ 0.5) "
            "| AUROC | Recall | FPR | TP | FP | FN |"
        ),
        "| --- " * 13 + "|",
    ]
    for r in table.to_dict("records"):
        name = "**Pooled**" if r["patient"] == "pooled" else r["patient"]
        lines.append(
            f"| {name} | {r['labelled_nuclei']:,} | {r['carcinoma_nuclei']:,} | {r['prevalence']:.4f} | "
            f"{r['AUPRC']:.4f} | {r['calibrated_AUPRC_pi0_pooled']:.4f} | {r['calibrated_AUPRC_pi0_0.5']:.4f} | "
            f"{r['AUROC']:.4f} | {r['recall_youden']:.4f} | {r['FPR_youden']:.4f} | {r['TP_youden']:,} | "
            f"{r['FP_youden']:,} | {r['FN_youden']:,} |"
        )
    lines += [
        "",
        "## Hard negatives (Another pathology)",
        "",
        "| Slide | Group | n | FP | FPR | Median score |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r.slide} | {r.group} | {r.n:,} | {r.FP_youden:,} | {r.FPR_youden:.4f} | {r.median_score:.4f} |"
        for r in hard_neg.itertuples()
    ]
    broken = order_tab[~order_tab["order_holds"]]
    lines += [
        "",
        "## Thesis order",
        "",
        "P1-P10 follow the descending prevalence with these labels."
        if broken.empty
        else "The thesis order breaks: "
        + ", ".join(
            f"{r.patient} is rank {r.rank_by_prevalence}" for r in broken.itertuples()
        )
        + ".",
        "",
        "## Checks",
        "",
        (
            "- Same nuclei, carcinoma counts, prevalence, per-patient AUPRC / AUROC and pooled per-seed metrics "
            "as the results run (max |Δ| "
            f"{max(v for k, v in checks['agreement_with_results_run'].items() if k.endswith('diff')):.1e})."
        ),
        (
            f"- Bootstrap replicates equal the results run's (max |Δ| AUPRC "
            f"{b['AUPRC']['max_abs_diff_replicates_vs_results_run']:.1e}, AUROC "
            f"{b['AUROC']['max_abs_diff_replicates_vs_results_run']:.1e})."
        ),
        (
            f"- Calibrated AUPRC with c = 1 equals the torchmetrics AUPRC (max |Δ| "
            f"{checks['calibration_check']['max_abs_diff_vs_torchmetrics']:.1e})."
        ),
        f"- Thresholds equal the params of runs `{config.quantiles_run}` and `{config.error_analysis_run}`.",
        "",
    ]
    return "\n".join(lines)


@with_cli_args(["+postprocessing=patient_analysis_another_negative_mmci"])
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

    b = checks["bootstrap"]
    params = {
        "results_run": config.results_run,
        "error_analysis_run": config.error_analysis_run,
        "quantiles_run": config.quantiles_run,
        "dataset": config.dataset_label,
        "another_pathology_at_test": "negative (labelled 0)",
        "threshold_youden": checks["thresholds"]["youden"],
        "threshold_recall_0.90": checks["thresholds"]["recall_0.90"],
        "pi0": ",".join(f"{k}={v}" for k, v in checks["pi0"].items()),
        "resamples_run": config.resamples_run,
        "resamples_sha256": b["resamples_sha256"],
        **{f"eval_run_{i}": r for i, r in enumerate(checks["eval_runs"], start=1)},
        **{
            f"training_run_{i}": r
            for i, r in enumerate(checks["training_runs"], start=1)
        },
    }
    tags = {
        "mlflow.runName": config.metadata.run_name,
        "mlflow.note.content": config.metadata.description,
        "results_run": config.results_run,
        "error_analysis_run": config.error_analysis_run,
        "quantiles_run": config.quantiles_run,
        **git_info(*CODE),
    }
    for text in [*map(str, tags.values()), *map(str, params.values())]:
        assert not CREDENTIAL_URL.search(text), "credential URL in a tag or param"
    experiment = client.get_experiment_by_name(config.metadata.experiment_name)
    run_id = client.create_run(
        experiment.experiment_id, run_name=config.metadata.run_name, tags=tags
    ).info.run_id
    user = client.get_run(config.results_run).data.tags.get("mlflow.user")
    if user:
        client.set_tag(run_id, "mlflow.user", user)
    for k, v in params.items():
        client.log_param(run_id, k, v)
    table = pd.read_csv(out / "patient_analysis_mmci.csv").set_index("patient")
    for col in table.columns:
        client.log_metric(run_id, f"pooled/{col}", float(table.loc["pooled", col]))
    for m in METRICS:
        client.log_metric(run_id, f"pooled/{m}_ci_lo", b[m]["ci_lo"])
        client.log_metric(run_id, f"pooled/{m}_ci_hi", b[m]["ci_hi"])
    for name in (
        "patient_analysis_mmci.csv",
        "patient_operating_points_mmci.csv",
        "patient_score_quantiles_mmci.csv",
        "hard_negatives_mmci.csv",
        "thesis_order_mmci.csv",
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
