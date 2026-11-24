# ruff: noqa: RUF001, RUF002  (Greek letters / minus signs in report text are intended)
r"""Case-balanced nucleus-level evaluation: pooled vs per-case weights.

Pooled metrics let a case (slide; MMCI patient) count in proportion to its number of labelled
nuclei. Here every case contributes equally within its class: nucleus i of class c on case s has
weight w_i = 1 / n(s, c), n(s, c) = labelled nuclei of class c on case s (a case without nuclei of a
class contributes nothing to it). With these weights:

- AUROC = Σ_{i∈pos, j∈neg} w_i w_j ([s_i > s_j] + ½[s_i = s_j]) / (W_pos W_neg);
- AUPRC = average precision with weighted TP / FP (Σ ΔTP_w / W_pos · TP_w / (TP_w + FP_w) over the
  distinct scores, as sklearn's ``average_precision_score`` with ``sample_weight``); W_pos and W_neg
  are the numbers of cases with nuclei of each class, so its prevalence is W_pos / (W_pos + W_neg);
- calibrated AUPRC (Siblini et al. 2020), π₀ = 0.5 and the prevalence of the weighted classes:
  precision TP_w / (TP_w + c·FP_w), c = W_pos / W_neg · (1 − π₀) / π₀, i.e. TPR / (TPR + FPR).

Point estimates are the seed means of the exact metrics. The bootstrap reuses the resamples of the
pooled versions (not redrawn): slides stratified by slide label (iCAIRD b973c17b and PANDA Radboud,
B = 2,000, seed 20261108) and MMCI patients (the tile-level resamples of e8682243, seed 20261114).
Each drawn copy of a case is a separate case with the weights above, so a replicate's class
histogram is Σ_s m_s / n(s, c) · h_s,c (m_s = multiplicity) and the weighted prevalence follows the
drawn cases. Per resample the seeds are averaged; percentile CIs; paired differences within a
resample, p = 2·min(P(Δ* ≤ 0), P(Δ* ≥ 0)); Holm per family over the AUPRC rows, as the pooled
versions (AUROC rows are supporting: raw p). Pooled values are recomputed by the same code and
compared with the reference runs (checks.json).

Replicates come from per-case logit histograms on each reference run's grid, doubled until every
run's full-test-set AUROC, AUPRC and calibrated AUPRC from the histograms match the exact values
within ``histogram.tolerance`` under both weightings.

Why values move (anonymous, ``auroc_drivers.csv``): ΔAUROC (balanced − pooled) is split exactly into
(1) balancing the positive cases (negatives pooled): Σ_s (1/K_pos − n_s/N_pos)·A_s, A_s = AUROC of
case s's carcinoma nuclei against the pooled negatives; (2) then balancing the negative cases:
−Σ_j (1/K_neg − n_j/N_neg)·U_j, U_j = P(a benign nucleus of case j outranks a carcinoma nucleus of
the balanced positives). Cases are grouped by size thirds within their class. The four weightings
(pooled, positives only, negatives only, both) are reported for every metric.

Sets: (a) iCAIRD test (sparse labels of b973c17b), Shape + RoVE and Position only + RoVE, shape −
position only; (b) PANDA Radboud test, main model, all and epithelial nuclei (as 027eeaee);
(c) MMCI TL test, Another pathology and Exclude ignored (e8682243 rule), patient = case;
(d) PANDA Radboud comparison families: sources (all: 4081f97a; epithelial: 44ffd165), context
(4081f97a / 027eeaee), positions (4081f97a).

Outputs (``out_dir``, logged to a new run in experiment 37): ``metrics.csv``, ``movement.csv``,
``comparisons.csv``, ``comparisons_side_by_side.csv``, ``case_counts.csv``, ``auroc_drivers.csv``,
``weighting_variants.csv``, ``checks.json``, ``report.md``, ``replicates.npz``. No slide, patient or
case ids in any output (checked before logging); the MMCI resamples are never logged.
"""

import json
import os
import re
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
from sklearn.metrics import average_precision_score, roc_auc_score

from postprocessing.bootstrap_sources import (
    bin_logits,
    bootstrap_p,
    holm,
    label_predictions,
    load_labels,
    load_run,
    slide_histograms,
    to_logits,
)
from postprocessing.icaird_test_results import SLIDE_ID, hist_all, sparse_labels
from postprocessing.nucleus_results_mmci import CREDENTIAL_URL, load_ignore
from postprocessing.results_context import epithelium_classes, load_slides
from postprocessing.tile.bootstrap_pixel_vs_nuclei import git_info


REPO = Path(__file__).parents[1]
CODE = ("postprocessing/case_balanced.py", "configs/postprocessing/case_balanced.yaml")
METRICS = ("AUPRC", "AUROC", "AUPRC_cal")
WEIGHTINGS = ("pooled", "balanced")
VARIANTS = ("pooled", "pos_balanced", "neg_balanced", "balanced")
NAMES = {
    "AUPRC": "AUPRC",
    "AUROC": "AUROC",
    "AUPRC_cal": "Cal. AUPRC (π₀ 0.5)",
    "AUPRC_prev": "AUPRC at the pooled prevalence",
}


# --------------------------------------------------------------------------
# Weights and exact metrics
# --------------------------------------------------------------------------


def class_counts(case: NDArray, y: NDArray, n_cases: int) -> tuple[NDArray, NDArray]:
    return (
        np.bincount(case[y == 1], minlength=n_cases).astype(np.float64),
        np.bincount(case[y == 0], minlength=n_cases).astype(np.float64),
    )


def inverse(n: NDArray) -> NDArray:
    with np.errstate(divide="ignore"):
        return np.where(n > 0, 1.0 / n, 0.0)


def metrics_from_steps(
    tp: NDArray, fp: NDArray, pi0: float, pi_ref: float | None = None
) -> dict[str, float]:
    """Metrics from cumulative weighted TP / FP at the distinct scores (descending).

    With ``pi_ref``, also ``AUPRC_prev``: the AP at prevalence ``pi_ref`` (the pooled nucleus
    prevalence), which separates the within-class reweighting from the change of prevalence.
    """
    w_pos, w_neg = tp[-1], fp[-1]
    d_tp, d_fp = np.diff(tp, prepend=0.0), np.diff(fp, prepend=0.0)

    def ap(pi_target: float) -> float:
        c = w_pos / w_neg * (1 - pi_target) / pi_target
        with np.errstate(invalid="ignore", divide="ignore"):
            prec = np.where(tp + fp > 0, tp / (tp + c * fp), 0.0)
        return float((d_tp * prec).sum() / w_pos)

    with np.errstate(invalid="ignore", divide="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    out = {
        "AUPRC": float((d_tp * prec).sum() / w_pos),
        "AUROC": float((d_tp * (w_neg - fp + 0.5 * d_fp)).sum() / (w_pos * w_neg)),
        "AUPRC_cal": ap(pi0),
    }
    if pi_ref is not None:
        out["AUPRC_prev"] = ap(pi_ref)
    return out


def exact_variants(
    y: NDArray, p: NDArray, case: NDArray, inv_p: NDArray, inv_n: NDArray, pi0: float
) -> dict[str, dict[str, float]]:
    pi_ref = float((y == 1).mean())
    """Exact metrics under the four weightings (pooled / balanced, per class)."""
    order = np.argsort(-p, kind="stable")
    ps, pos, cs = p[order], y[order] == 1, case[order]
    last = np.r_[np.flatnonzero(ps[1:] != ps[:-1]), len(ps) - 1]
    del ps
    wp = {"pooled": pos.astype(np.float64), "balanced": np.where(pos, inv_p[cs], 0.0)}
    wn = {
        "pooled": (~pos).astype(np.float64),
        "balanced": np.where(~pos, inv_n[cs], 0.0),
    }
    tp = {k: np.cumsum(v)[last] for k, v in wp.items()}
    fp = {k: np.cumsum(v)[last] for k, v in wn.items()}
    return {
        "pooled": metrics_from_steps(tp["pooled"], fp["pooled"], pi0, pi_ref),
        "pos_balanced": metrics_from_steps(tp["balanced"], fp["pooled"], pi0, pi_ref),
        "neg_balanced": metrics_from_steps(tp["pooled"], fp["balanced"], pi0, pi_ref),
        "balanced": metrics_from_steps(tp["balanced"], fp["balanced"], pi0, pi_ref),
    }


def sklearn_check(y: NDArray, p: NDArray, w: NDArray, mine: dict[str, float]) -> dict:
    """Weighted AUPRC / AUROC against sklearn with sample_weight."""
    ref = {
        "AUPRC": float(average_precision_score(y, p, sample_weight=w)),
        "AUROC": float(roc_auc_score(y, p, sample_weight=w)),
    }
    return {
        m: {"sklearn": ref[m], "here": mine[m], "abs_diff": abs(ref[m] - mine[m])}
        for m in ref
    }


# --------------------------------------------------------------------------
# Histograms, replicates, AUROC drivers
# --------------------------------------------------------------------------


def grid_errors(
    probs: NDArray,
    y: NDArray,
    case: NDArray,
    inv_p: NDArray,
    inv_n: NDArray,
    exact: list[dict],
    lo: float,
    hi: float,
    n_bins: int,
    pi0: float,
) -> float:
    pos, neg = y == 1, y == 0
    w_p, w_n = inv_p[case[pos]], inv_n[case[neg]]
    worst = 0.0
    for p, e in zip(probs, exact, strict=True):
        b = bin_logits(to_logits(p), lo, hi, n_bins)
        for w, (hp, hn) in {
            "pooled": (
                np.bincount(b[pos], minlength=n_bins),
                np.bincount(b[neg], minlength=n_bins),
            ),
            "balanced": (
                np.bincount(b[pos], weights=w_p, minlength=n_bins),
                np.bincount(b[neg], weights=w_n, minlength=n_bins),
            ),
        }.items():
            h = hist_all(hp.astype(np.float64), hn.astype(np.float64), pi0)
            worst = max(worst, *(abs(float(h[m]) - e[w][m]) for m in METRICS))
    return worst


def replicates(
    pos: NDArray,
    neg: NDArray,
    mult: NDArray,
    inv_p: NDArray,
    inv_n: NDArray,
    pi0: float,
    chunk: int = 250,
) -> dict[str, dict[str, NDArray]]:
    out: dict = {w: {m: [] for m in METRICS} for w in WEIGHTINGS}
    for start in range(0, len(mult), chunk):
        m = mult[start : start + chunk].astype(np.float64)
        for w, (a, b) in {"pooled": (m, m), "balanced": (m * inv_p, m * inv_n)}.items():
            for k, v in hist_all(a @ pos, b @ neg, pi0).items():
                out[w][k].append(v)
    return {w: {m: np.concatenate(v) for m, v in d.items()} for w, d in out.items()}


def auroc_drivers(
    pos: NDArray, neg: NDArray, inv_p: NDArray, inv_n: NDArray
) -> dict[str, NDArray]:
    """Per-case terms of ΔAUROC (balanced − pooled), two exact steps; see the module docstring."""
    n_pos, n_neg = pos.sum(1), neg.sum(1)
    neg_pool = neg.sum(0)
    below = np.cumsum(neg_pool) - neg_pool + 0.5 * neg_pool
    with np.errstate(invalid="ignore", divide="ignore"):
        a = (pos @ below) / (
            n_pos * neg_pool.sum()
        )  # case's carcinoma vs pooled negatives
    pos_bal = inv_p @ pos
    pos_below = (
        np.cumsum(pos_bal) - pos_bal + 0.5 * pos_bal
    )  # carcinoma below + half ties
    with np.errstate(invalid="ignore", divide="ignore"):
        u = (neg @ pos_below) / (
            n_neg * pos_bal.sum()
        )  # P(case's benign nucleus outranks a positive)
    has_p, has_n = n_pos > 0, n_neg > 0
    k_p, k_n = has_p.sum(), has_n.sum()
    step1 = np.where(has_p, (1 / k_p - n_pos / n_pos.sum()) * np.nan_to_num(a), 0.0)
    step2 = np.where(has_n, -(1 / k_n - n_neg / n_neg.sum()) * np.nan_to_num(u), 0.0)
    return {
        "A": a,
        "U": u,
        "step1": step1,
        "step2": step2,
        "auroc_pooled": float(np.nansum(a * n_pos) / n_pos.sum()),
        "auroc_pos_balanced": float(np.nanmean(a[has_p])),
        "auroc_balanced": float(1 - np.nanmean(u[has_n])),
    }


def evaluate(
    label: str,
    runs: list[tuple[str, int, str]],
    probs: NDArray,
    y: NDArray,
    case: NDArray,
    n_cases: int,
    mult: NDArray,
    grid: tuple[float, float, int],
    config: DictConfig,
    other_class: NDArray | None = None,
) -> dict:
    """Exact metrics (four weightings), validated grid, replicates and AUROC drivers of one set."""
    pi0 = float(config.pi0_reference)
    n_p, n_n = class_counts(case, y, n_cases)
    inv_p, inv_n = inverse(n_p), inverse(n_n)
    exact = []
    for i, p in enumerate(probs):
        exact.append(exact_variants(y, p, case, inv_p, inv_n, pi0))
        print(
            f"[{label}] {runs[i][0]} seed {runs[i][1]}: exact {json.dumps(exact[-1]['balanced'])}",
            flush=True,
        )
    w = np.where(y == 1, inv_p[case], inv_n[case])
    sk = sklearn_check(y, probs[0], w, exact[0]["balanced"])
    sk_pooled = sklearn_check(y, probs[0], np.ones(len(y)), exact[0]["pooled"])
    del w

    lo, hi, n_bins = grid
    n_bins0 = n_bins
    for _ in range(int(config.histogram.max_doublings) + 1):
        err = grid_errors(probs, y, case, inv_p, inv_n, exact, lo, hi, n_bins, pi0)
        print(f"[{label}] grid {n_bins} bins: max |hist − exact| {err:.2e}", flush=True)
        if err <= float(config.histogram.tolerance):
            break
        n_bins *= 2
    assert err <= float(config.histogram.tolerance), f"{label}: grid too coarse"

    rep = {
        wt: {m: np.empty((len(probs), len(mult))) for m in METRICS} for wt in WEIGHTINGS
    }
    drivers = []
    for i, p in enumerate(probs):
        pos, neg = slide_histograms(
            case, bin_logits(to_logits(p), lo, hi, n_bins), y, n_cases, n_bins
        )
        for wt, d in replicates(pos, neg, mult, inv_p, inv_n, pi0).items():
            for m, v in d.items():
                rep[wt][m][i] = v
        drivers.append(auroc_drivers(pos, neg, inv_p, inv_n))
        del pos, neg
    for wt in WEIGHTINGS:
        assert all(np.isfinite(rep[wt][m]).all() for m in METRICS), (
            f"{label}: non-finite replicates"
        )
    # the driver identities hold on the histograms (ties within a bin)
    driver_err = max(
        max(
            abs(d["auroc_pooled"] + d["step1"].sum() - d["auroc_pos_balanced"]),
            abs(d["auroc_pos_balanced"] + d["step2"].sum() - d["auroc_balanced"]),
            abs(d["auroc_pooled"] - e["pooled"]["AUROC"]),
            abs(d["auroc_balanced"] - e["balanced"]["AUROC"]),
            abs(d["auroc_pos_balanced"] - e["pos_balanced"]["AUROC"]),
        )
        for d, e in zip(drivers, exact, strict=True)
    )
    assert driver_err <= float(config.histogram.tolerance), (
        f"{label}: AUROC drivers {driver_err:.2e}"
    )
    return {
        "runs": runs,
        "exact": exact,
        "rep": rep,
        "drivers": drivers,
        "n_pos": n_p,
        "n_neg": n_n,
        "other_class": other_class,
        "grid": {
            "logit_range": [lo, hi],
            "n_bins_reference": n_bins0,
            "n_bins": n_bins,
            "max_abs_error": err,
            "auroc_driver_max_abs_error": driver_err,
        },
        "sklearn_check_first_run": {"balanced": sk, "pooled": sk_pooled},
        "counts": {
            "nuclei": len(y),
            "carcinoma_nuclei": int((y == 1).sum()),
            "benign_nuclei": int((y == 0).sum()),
            "cases_with_carcinoma": int((n_p > 0).sum()),
            "cases_with_benign": int((n_n > 0).sum()),
            "prevalence_pooled": float((y == 1).mean()),
            "prevalence_balanced": float(
                (n_p > 0).sum() / ((n_p > 0).sum() + (n_n > 0).sum())
            ),
        },
    }


# --------------------------------------------------------------------------
# Summaries
# --------------------------------------------------------------------------


def model_indices(res: dict) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for i, (name, _, _) in enumerate(res["runs"]):
        out.setdefault(name, []).append(i)
    return out


def summarise(
    set_label: str, subset: str, res: dict, labels: dict, config: DictConfig
) -> dict:
    """Seed-mean points, CIs, movement; seed-averaged replicates per model."""
    q = (1 - float(config.metrics_ci_level)) / 2 * 100
    avg_rep: dict = {wt: {m: {} for m in METRICS} for wt in WEIGHTINGS}
    avg_pt: dict = {v: {m: {} for m in METRICS} for v in VARIANTS}
    rows, moves, variants = [], [], []
    for name, idx in model_indices(res).items():
        for m in METRICS:
            for v in VARIANTS:
                avg_pt[v][m][name] = float(
                    np.mean([res["exact"][i][v][m] for i in idx])
                )
            for wt in WEIGHTINGS:
                r = res["rep"][wt][m][idx].mean(0)
                avg_rep[wt][m][name] = r
                seeds = np.array([res["exact"][i][wt][m] for i in idx])
                lo, hi = np.percentile(r, [q, 100 - q])
                rows.append(
                    {
                        "set": set_label,
                        "nuclei": subset,
                        "model": name,
                        "label": labels[name],
                        "metric": m,
                        "weighting": wt,
                        "point": float(seeds.mean()),
                        "ci_lo": float(lo),
                        "ci_hi": float(hi),
                        **{f"seed_{k + 1}": float(s) for k, s in enumerate(seeds)},
                        "seed_sd": float(seeds.std(ddof=1)),
                    }
                )
            d = avg_rep["balanced"][m][name] - avg_rep["pooled"][m][name]
            lo, hi = np.percentile(d, [q, 100 - q])
            moves.append(
                {
                    "set": set_label,
                    "nuclei": subset,
                    "model": name,
                    "metric": m,
                    "pooled": avg_pt["pooled"][m][name],
                    "balanced": avg_pt["balanced"][m][name],
                    "change": avg_pt["balanced"][m][name] - avg_pt["pooled"][m][name],
                    "change_ci_lo": float(lo),
                    "change_ci_hi": float(hi),
                    "change_p": bootstrap_p(d),
                }
            )
        for m in (*METRICS, "AUPRC_prev"):
            variants.append(
                {
                    "set": set_label,
                    "nuclei": subset,
                    "model": name,
                    "metric": m,
                    **{
                        v: float(np.mean([res["exact"][i][v][m] for i in idx]))
                        for v in VARIANTS
                    },
                }
            )
    return {
        "rows": rows,
        "moves": moves,
        "variants": variants,
        "avg_rep": avg_rep,
        "avg_pt": avg_pt,
    }


def compare(
    set_label: str,
    subset: str,
    family: str,
    comparisons: list,
    summ: dict,
    config: DictConfig,
    holm_family: bool,
) -> list[dict]:
    q = (1 - float(config.metrics_ci_level)) / 2 * 100
    out = []
    for wt in WEIGHTINGS:
        for m in METRICS:
            rows = []
            for x in comparisons:
                diff = sum(w * summ["avg_rep"][wt][m][n] for n, w in x.weights.items())
                point = sum(w * summ["avg_pt"][wt][m][n] for n, w in x.weights.items())
                lo, hi = np.percentile(diff, [q, 100 - q])
                rows.append(
                    {
                        "set": set_label,
                        "nuclei": subset,
                        "family": family,
                        "comparison": x.name,
                        "label": x.label,
                        "metric": m,
                        "weighting": wt,
                        "difference": float(point),
                        "ci_lo": float(lo),
                        "ci_hi": float(hi),
                        "ci_excludes_zero": bool(lo > 0 or hi < 0),
                        "p": bootstrap_p(diff),
                        "p_holm": np.nan,
                        "significant": None,
                        "report": bool(x.get("report", False)),
                    }
                )
            if holm_family and m == "AUPRC":
                for r, a in zip(
                    rows, holm(np.array([r["p"] for r in rows])), strict=True
                ):
                    r["p_holm"] = float(a)
                    r["significant"] = "yes" if a < float(config.alpha) else "no"
            out += rows
    return out


def side_by_side(comp: pd.DataFrame, alpha: float) -> pd.DataFrame:
    keys = ["set", "nuclei", "family", "comparison", "label", "metric", "report"]
    a = comp[comp["weighting"] == "pooled"].drop(columns="weighting")
    b = comp[comp["weighting"] == "balanced"].drop(columns="weighting")
    m = a.merge(b, on=keys, suffixes=("_pooled", "_balanced"), validate="1:1")

    def sig(
        r: pd.Series, s: str
    ) -> bool:  # Holm for AUPRC rows of a Holm family, raw p otherwise
        return (
            bool(r[f"significant_{s}"] == "yes")
            if pd.notna(r[f"p_holm_{s}"])
            else bool(r[f"p_{s}"] < alpha)
        )

    m["sign_change"] = np.sign(m["difference_pooled"]) != np.sign(
        m["difference_balanced"]
    )
    m["significant_pooled_any"] = m.apply(lambda r: sig(r, "pooled"), axis=1)
    m["significant_balanced_any"] = m.apply(lambda r: sig(r, "balanced"), axis=1)
    m["significance_change"] = (
        m["significant_pooled_any"] != m["significant_balanced_any"]
    )
    m["flag"] = [
        "+".join(f for f, on in (("SIGN", a), ("SIGNIFICANCE", b)) if on)
        for a, b in zip(m["sign_change"], m["significance_change"], strict=True)
    ]
    return m


def case_count_rows(
    set_label: str, subset: str, res: dict, case_unit: str
) -> list[dict]:
    rows = []
    for cls, n in (("carcinoma", res["n_pos"]), ("benign", res["n_neg"])):
        v = n[n > 0]
        share = v / v.sum()
        rows.append(
            {
                "set": set_label,
                "nuclei": subset,
                "case": case_unit,
                "nucleus_class": cls,
                "cases": len(v),
                "nuclei_total": int(v.sum()),
                "min": int(v.min()),
                "median": float(np.median(v)),
                "max": int(v.max()),
                "largest_case_share": float(share.max()),
                "largest_10pct_cases_share": float(
                    np.sort(share)[::-1][: max(1, round(0.1 * len(v)))].sum()
                ),
                "effective_cases_pooled": float(
                    1 / (share**2).sum()
                ),  # Kish: 1 / Σ w², pooled weights
            }
        )
    return rows


def driver_rows(set_label: str, subset: str, model: str, res: dict) -> list[dict]:
    """ΔAUROC contributions by size third within each class; seed means over the model's runs."""
    idx = model_indices(res)[model]
    d = res["drivers"]
    a = np.nanmean([d[i]["A"] for i in idx], axis=0)
    u = np.nanmean([d[i]["U"] for i in idx], axis=0)
    s1 = np.mean([d[i]["step1"] for i in idx], axis=0)
    s2 = np.mean([d[i]["step2"] for i in idx], axis=0)
    rows = []
    for cls, n, score, contrib, step in (
        ("carcinoma", res["n_pos"], a, s1, "1: balance carcinoma cases"),
        ("benign", res["n_neg"], u, s2, "2: balance benign cases"),
    ):
        has = np.flatnonzero(n > 0)
        order = has[np.argsort(n[has], kind="stable")]
        groups = [
            (f"size third {k + 1} (smallest first)", g)
            for k, g in enumerate(np.array_split(order, 3))
        ]
        groups.append(("largest case", order[-1:]))
        if cls == "benign" and res["other_class"] is not None:
            oc = res["other_class"]
            groups += [
                ("cases with carcinoma", has[oc[has]]),
                ("cases without carcinoma", has[~oc[has]]),
            ]
        for name, g in groups:
            if len(g) == 0:
                continue
            rows.append(
                {
                    "set": set_label,
                    "nuclei": subset,
                    "model": model,
                    "nucleus_class": cls,
                    "step": step,
                    "group": name,
                    "cases": len(g),
                    "nuclei_min": int(n[g].min()),
                    "nuclei_max": int(n[g].max()),
                    "pooled_weight": float(n[g].sum() / n[has].sum()),
                    "balanced_weight": len(g) / len(has),
                    "score_name": "A (AUROC vs pooled benign)"
                    if cls == "carcinoma"
                    else "U (P benign outranks carcinoma)",
                    "mean_score": float(np.mean(score[g])),
                    "contribution_to_delta_auroc": float(contrib[g].sum()),
                }
            )
    return rows


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def set_tracking(uri: str) -> None:
    os.environ["MLFLOW_TRACKING_URI"] = uri
    mlflow.set_tracking_uri(uri)


def load_icaird(c: DictConfig) -> dict:
    slides = pd.read_parquet(download_artifacts(c.metadata_uri)).sort_values(
        "slide_id", ignore_index=True
    )
    slide_ids = slides["slide_id"].tolist()
    runs, probs, ref = [], [], None
    for name, m in c.models.items():
        for seed, eval_id in enumerate(m.eval_runs, start=1):
            repl = (m.get("replacements") or {}).get(eval_id)
            if repl is not None:
                df = pd.read_parquet(repl.test_predictions)
            else:
                df, info = load_run(
                    eval_id, slide_ids, Path(c.cache_dir), c.predictions_artifact
                )
                assert (
                    info["test_slides_without_predictions"] == 0
                    and info["files_outside_test_metadata"] == 0
                )
            if ref is None:
                ref = df[["slide", "key"]]
            assert np.array_equal(df["slide"].to_numpy(), ref["slide"].to_numpy())
            assert np.array_equal(df["key"].to_numpy(), ref["key"].to_numpy()), (
                "different nuclei"
            )
            runs.append((name, seed, eval_id))
            probs.append(df["p"].to_numpy(np.float32))
    y_all, _ = sparse_labels(slides, ref, Path(c.supervision_dir), c.label_column)
    keep = y_all >= 0
    case = ref["slide"].to_numpy()[keep].astype(np.int64)
    z = np.load(c.resamples)
    positive = slides["is_carcinoma"].to_numpy(bool)
    assert np.array_equal(z["positive"], positive), "iCAIRD resamples: other strata"
    assert int(z["seed"]) == 20261108
    ref_checks = json.loads((Path(c.reference_dir) / "checks.json").read_text())[
        "histogram"
    ]
    lo, hi = ref_checks["shape"]["logit_range"]
    n_bins = int(ref_checks["shape"]["n_bins"])
    assert all(
        int(v["n_bins"]) == n_bins and v["logit_range"] == [lo, hi]
        for v in ref_checks.values()
    )
    return {
        "runs": runs,
        "probs": np.stack(probs)[:, keep],
        "y": y_all[keep],
        "case": case,
        "n_cases": len(slides),
        "mult": z["multiplicities"],
        "grid": (lo, hi, n_bins),
        "private_ids": slide_ids,
        "other_class": None,
    }


def load_radboud(c: DictConfig, names: list[str]) -> dict:
    slides = load_slides(c.metadata_uri)
    slide_ids = slides["slide_id"].tolist()
    labels = load_labels(slides, Path(c.supervision_dir), c.label_column)
    carcinoma = slides["is_carcinoma"].to_numpy(bool)
    runs, probs, ref, y = [], [], None, None
    for name in names:
        for seed, run_id in enumerate(c.configurations[name].eval_runs, start=1):
            df, _ = load_run(
                run_id, slide_ids, Path(c.cache_dir), c.predictions_artifact
            )
            yy, _ = label_predictions(df, labels, carcinoma)
            if ref is None:
                ref, y = df[["slide", "key"]], yy
            assert np.array_equal(df["slide"].to_numpy(), ref["slide"].to_numpy())
            assert np.array_equal(df["key"].to_numpy(), ref["key"].to_numpy()), (
                f"{run_id}: other nuclei"
            )
            assert np.array_equal(yy, y)
            runs.append((name, seed, run_id))
            probs.append(df["p"].to_numpy(np.float32))
            print(f"radboud {name} seed {seed}: loaded", flush=True)
    slide = ref["slide"].to_numpy().astype(np.int64)
    cls, _ = epithelium_classes(c, slides, slide, y)
    epi = (y == 1) | (cls == 2)
    del cls
    z = np.load(c.resamples)
    assert z["slide_ids"].tolist() == slide_ids, "Radboud resamples: other slides"
    positive = np.bincount(slide, weights=(y == 1), minlength=len(slides)) > 0
    assert np.array_equal(z["positive"], positive), "Radboud resamples: other strata"
    assert int(z["seed"]) == 20261108
    ref_hist = json.loads((Path(c.reference_all_dir) / "checks.json").read_text())[
        "histogram"
    ]
    lo, hi = ref_hist["logit_range_q2"]
    return {
        "runs": runs,
        "probs": np.stack(probs),
        "y": y,
        "case": slide,
        "epithelium": epi,
        "n_cases": len(slides),
        "mult": z["multiplicities"],
        "grid": (lo, hi, int(ref_hist["n_bins"])),
        "private_ids": slide_ids,
        "other_class": positive,
    }


def load_mmci(c: DictConfig, sensitive_uri: str) -> dict:
    """MMCI predictions, final test rule, patient resamples (read with the sensitive server)."""
    set_tracking(sensitive_uri)
    meta = pd.read_parquet(download_artifacts(c.metadata_uri)).sort_values(
        "slide_id", ignore_index=True
    )
    meta["patient_id"] = meta["patient_id"].astype(str)
    slides = meta[["slide_id", "is_carcinoma"]]
    slide_ids = slides["slide_id"].tolist()
    carcinoma = slides["is_carcinoma"].to_numpy(bool)
    labels = load_labels(slides, Path(c.supervision_dir), c.label_column)
    runs, probs, ref = [], [], None
    for seed, run_id in enumerate(c.eval_runs, start=1):
        df, info = load_run(
            run_id, slide_ids, Path(c.cache_dir), c.predictions_artifact
        )
        assert (
            info["test_slides_without_predictions"] == 0
            and info["files_outside_test_metadata"] == 0
        )
        if ref is None:
            ref = df[["slide", "key"]]
        assert np.array_equal(df["key"].to_numpy(), ref["key"].to_numpy()), (
            "seeds cover different nuclei"
        )
        runs.append(("mmci", seed, run_id))
        probs.append(df["p"].to_numpy(np.float32))
    y, _ = label_predictions(ref, labels, carcinoma)
    ign = load_ignore(slides, Path(c.test_ignore_dir))
    merged = ref.merge(ign, on=["slide", "key"], how="left", validate="1:1")
    ignore = merged["ignore"].eq(True).to_numpy()
    assert int(ignore.sum()) == len(ign) and int((y[ignore] == 1).sum()) == 0
    keep = ~ignore
    res_path = Path(
        download_artifacts(run_id=c.resamples_run, artifact_path=c.resamples_artifact)
    )
    r = np.load(res_path)
    assert int(r["seed"]) == int(c.resamples_seed)
    patients = [str(p) for p in r["patients"]]
    assert sorted(patients) == sorted(meta["patient_id"].unique())
    pidx = {p: i for i, p in enumerate(patients)}
    slide_patient = meta["patient_id"].map(pidx).to_numpy()
    case = slide_patient[ref["slide"].to_numpy()[keep]].astype(np.int64)
    mapping = pd.read_csv(c.patient_label_mapping, dtype=str)
    ref_hist = json.loads((Path(c.reference_dir) / "checks.json").read_text())[
        "histogram"
    ]
    lo, hi = ref_hist["logit_range"]
    y = y[keep]
    has_carcinoma = np.bincount(case, weights=(y == 1), minlength=len(patients)) > 0
    out = {
        "runs": runs,
        "probs": np.stack(probs)[:, keep],
        "y": y,
        "case": case,
        "n_cases": len(patients),
        "mult": r["multiplicities"],
        "grid": (lo, hi, int(ref_hist["n_bins"])),
        "private_ids": [
            *mapping["patient_id"],
            *mapping["case_id"],
            *slide_ids,
            *patients,
        ],
        "other_class": has_carcinoma,
        "ignored": int(ignore.sum()),
    }
    del r
    return out


# --------------------------------------------------------------------------
# Reference checks (pooled values reproduce the earlier runs)
# --------------------------------------------------------------------------


def max_abs(a: NDArray, b: NDArray) -> float:
    return float(np.abs(np.asarray(a, float) - np.asarray(b, float)).max())


def check_icaird(c: DictConfig, res: dict, summ: dict, comp: pd.DataFrame) -> dict:
    d = Path(c.reference_dir)
    z = np.load(d / "replicates.npz")
    idx = model_indices(res)
    rep = {
        f"{k}/{m}": max_abs(res["rep"]["pooled"][m][idx[k]], z[f"{k}/{m}"])
        for k in idx
        for m in METRICS
    }
    nr = pd.read_csv(d / "nucleus_results_icaird.csv")
    pts = {}
    for r in nr.itertuples():
        k = "shape" if r.model.startswith("Shape") else "position"  # b973c17b labels
        for m in METRICS:
            mine = next(
                x
                for x in summ["rows"]
                if x["model"] == k and x["metric"] == m and x["weighting"] == "pooled"
            )
            pts[f"{k}/{m}"] = max_abs(
                [mine["point"], mine["ci_lo"], mine["ci_hi"]],
                [getattr(r, m), getattr(r, f"{m}_ci_lo"), getattr(r, f"{m}_ci_hi")],
            )
    ladder = pd.read_csv(d / "ladder_icaird.csv").set_index("metric")
    lad = {}
    for m in METRICS:
        mine = comp[(comp["weighting"] == "pooled") & (comp["metric"] == m)].iloc[0]
        lad[m] = max_abs(
            [mine["difference"], mine["ci_lo"], mine["ci_hi"]],
            ladder.loc[m, ["delta_shape_minus_position", "ci_lo", "ci_hi"]].to_numpy(),
        )
    lad["p_AUPRC"] = abs(
        float(
            comp[(comp["weighting"] == "pooled") & (comp["metric"] == "AUPRC")][
                "p"
            ].iloc[0]
        )
        - float(ladder.loc["AUPRC", "p_bootstrap"])
    )
    return {
        "run": "b973c17b",
        "max_abs_replicate_difference": rep,
        "max_abs_point_ci_difference": pts,
        "max_abs_ladder_difference": lad,
    }


def check_radboud(
    c: DictConfig, res: dict, subset: str, summ: dict, comp: pd.DataFrame
) -> dict:
    out: dict = {}
    refs = {
        "all": [("4081f97a", Path(c.reference_all_dir), "{m}", None)],
        "epithelium": [
            (
                "027eeaee",
                Path(c.reference_context_dir),
                "epithelium/{m}",
                "context_epithelium",
            ),
            (
                "44ffd165",
                Path(c.reference_sources_epithelium_dir),
                "epithelium/{m}",
                "sources_epithelium",
            ),
        ],
    }[subset]
    run_ids = [r for _, _, r in res["runs"]]
    for name, d, key, fam in refs:
        with np.load(d / "replicates.npz") as z:
            ref_runs = z["eval_runs"].tolist()
            shared = [
                (i, ref_runs.index(r)) for i, r in enumerate(run_ids) if r in ref_runs
            ]
            rep = {
                m: max(
                    max_abs(res["rep"]["pooled"][m][i], z[key.format(m=m)][j])
                    for i, j in shared
                )
                for m in ("AUPRC", "AUROC")
            }
        rc = pd.read_csv(d / "comparisons.csv")
        if "subset" in rc:
            rc = rc[rc["subset"] == subset]
        mine = comp[
            (comp["weighting"] == "pooled")
            & (comp["nuclei"] == subset)
            & comp["metric"].isin(["AUPRC", "AUROC"])
        ]
        if fam is not None:
            rc = rc[rc["family"] == fam]
            mine = mine[mine["family"] == fam.split("_")[0]]
        mm = mine.merge(rc, on=["comparison", "metric"], suffixes=("", "_ref"))
        holm_rows = mm[mm["metric"] == "AUPRC"]
        out[name] = {
            "shared_runs": len(shared),
            "max_abs_replicate_difference": rep,
            "comparisons_matched": len(mm),
            "max_abs_comparison_difference": max_abs(
                mm[["difference", "ci_lo", "ci_hi", "p"]].to_numpy(),
                mm[["difference_ref", "ci_lo_ref", "ci_hi_ref", "p_ref"]].to_numpy(),
            ),
            "max_abs_p_holm_difference": max_abs(
                holm_rows["p_holm"], holm_rows["p_holm_ref"]
            ),
            "significance_identical": bool(
                (holm_rows["significant"] == holm_rows["significant_ref"]).all()
            ),
        }
    if subset == "all":
        nr = pd.read_csv(c.reference_nucleus_results).iloc[0]
        rows = [
            x
            for x in summ["rows"]
            if x["model"] == c.main_model and x["weighting"] == "pooled"
        ]
        out["tab_res_nucleus"] = {
            m: max_abs(
                [r["point"], r["ci_lo"], r["ci_hi"]],
                [nr[m], nr[f"{m}_ci_lo"], nr[f"{m}_ci_hi"]],
            )
            for r in rows
            for m in ("AUPRC", "AUROC")
            if r["metric"] == m
        }
    return out


def check_mmci(c: DictConfig, res: dict, summ: dict) -> dict:
    d = Path(c.reference_dir)
    z = np.load(d / "replicates.npz")
    assert z["eval_runs"].tolist() == list(c.eval_runs)
    rep = {m: max_abs(res["rep"]["pooled"][m], z[m]) for m in ("AUPRC", "AUROC")}
    nr = pd.read_csv(d / "nucleus_results_mmci.csv").iloc[0]
    rows = [x for x in summ["rows"] if x["weighting"] == "pooled"]
    pts = {
        r["metric"]: max_abs(
            [r["point"], r["ci_lo"], r["ci_hi"]],
            [nr[r["metric"]], nr[f"{r['metric']}_ci_lo"], nr[f"{r['metric']}_ci_hi"]],
        )
        for r in rows
        if r["metric"] in ("AUPRC", "AUROC")
    }
    return {
        "run": "e8682243",
        "max_abs_replicate_difference": rep,
        "max_abs_point_ci_difference": pts,
        "labelled_nuclei_equal": int(nr["labeled_nuclei"])
        == int(res["counts"]["nuclei"]),
    }


# --------------------------------------------------------------------------
# Main computation
# --------------------------------------------------------------------------


def compute(config: DictConfig) -> tuple[Path, dict, list[str]]:
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, moves, variants, comps, counts, drivers = [], [], [], [], [], []
    checks: dict = {"pi0_reference": float(config.pi0_reference), "sets": {}}
    private: list[str] = []
    replicate_store: dict[str, NDArray] = {}

    def store(prefix: str, res: dict) -> None:
        for wt in WEIGHTINGS:
            for m in METRICS:
                replicate_store[f"{prefix}/{wt}/{m}"] = res["rep"][wt][m].astype(
                    np.float32
                )
        replicate_store[f"{prefix}/eval_runs"] = np.array(
            [r for _, _, r in res["runs"]]
        )

    # (c) MMCI first (sensitive server), then everything on the public server ---------------
    mc = load_mmci(config.mmci, config.mlflow_sensitive_uri)
    set_tracking(config.mlflow_tracking_uri)
    private += mc.pop("private_ids")
    ignored = mc.pop("ignored")
    res = evaluate(
        "MMCI",
        mc["runs"],
        mc["probs"],
        mc["y"],
        mc["case"],
        mc["n_cases"],
        mc["mult"],
        mc["grid"],
        config,
        mc["other_class"],
    )
    del mc
    s = summarise("MMCI", "all", res, {"mmci": "Shape + RoVE (k = 44)"}, config)
    rows += s["rows"]
    moves += s["moves"]
    variants += s["variants"]
    counts += case_count_rows("MMCI", "all", res, "patient")
    drivers += driver_rows("MMCI", "all", "mmci", res)
    chk = check_mmci(config.mmci, res, s)
    assert chk["labelled_nuclei_equal"], "MMCI: labelled nuclei differ from e8682243"
    checks["sets"]["MMCI"] = {
        **chk,
        "counts": res["counts"],
        "ignored_nuclei": ignored,
        "grid": res["grid"],
        "sklearn": res["sklearn_check_first_run"],
        "resamples": {
            "run": config.mmci.resamples_run,
            "seed": int(config.mmci.resamples_seed),
            "unit": "patient",
            "relogged": False,
        },
    }
    store("mmci", res)
    del res

    # (a) iCAIRD -------------------------------------------------------------------------------
    ic = load_icaird(config.icaird)
    private += ic.pop("private_ids")
    res = evaluate(
        "iCAIRD",
        ic["runs"],
        ic["probs"],
        ic["y"],
        ic["case"],
        ic["n_cases"],
        ic["mult"],
        ic["grid"],
        config,
        ic["other_class"],
    )
    del ic
    labels = {k: v.label for k, v in config.icaird.models.items()}
    s = summarise("iCAIRD", "all", res, labels, config)
    rows += s["rows"]
    moves += s["moves"]
    variants += s["variants"]
    c_ic = compare(
        "iCAIRD",
        "all",
        "ladder",
        config.icaird.comparisons,
        s,
        config,
        holm_family=False,
    )
    comps += c_ic
    counts += case_count_rows("iCAIRD", "all", res, "slide")
    for k in labels:
        drivers += driver_rows("iCAIRD", "all", k, res)
    checks["sets"]["iCAIRD"] = {
        **check_icaird(config.icaird, res, s, pd.DataFrame(c_ic)),
        "counts": res["counts"],
        "grid": res["grid"],
        "sklearn": res["sklearn_check_first_run"],
    }
    store("icaird", res)
    del res

    # (b), (d) PANDA Radboud -------------------------------------------------------------------
    rc = config.radboud
    fams = rc.families
    names_all = list(rc.configurations)
    rb = load_radboud(rc, names_all)
    private += rb.pop("private_ids")
    labels = {k: v.label for k, v in rc.configurations.items()}
    checks["sets"]["PANDA Radboud"] = {}
    for subset in ("all", "epithelium"):
        needed = sorted(
            {
                n
                for f in fams.values()
                if subset in f.subsets
                for x in f.comparisons
                for n in x.weights
            }
            | {rc.main_model},
            key=names_all.index,
        )
        sel_runs = [i for i, (n, _, _) in enumerate(rb["runs"]) if n in needed]
        mask = rb["epithelium"] if subset == "epithelium" else slice(None)
        res = evaluate(
            f"Radboud {subset}",
            [rb["runs"][i] for i in sel_runs],
            rb["probs"][sel_runs][:, mask],
            rb["y"][mask],
            rb["case"][mask],
            rb["n_cases"],
            rb["mult"],
            rb["grid"],
            config,
            rb["other_class"],
        )
        s = summarise("PANDA Radboud", subset, res, labels, config)
        rows += [r for r in s["rows"] if r["model"] == rc.main_model]
        moves += s["moves"]
        variants += s["variants"]
        c_rb = []
        for fname, f in fams.items():
            if subset in f.subsets:
                c_rb += compare(
                    "PANDA Radboud",
                    subset,
                    fname,
                    f.comparisons,
                    s,
                    config,
                    holm_family=True,
                )
        comps += c_rb
        counts += case_count_rows("PANDA Radboud", subset, res, "slide")
        drivers += driver_rows("PANDA Radboud", subset, rc.main_model, res)
        checks["sets"]["PANDA Radboud"][subset] = {
            "reference": check_radboud(rc, res, subset, s, pd.DataFrame(c_rb)),
            "counts": res["counts"],
            "grid": res["grid"],
            "sklearn": res["sklearn_check_first_run"],
        }
        store(f"radboud_{subset}", res)
        del res
    del rb

    # outputs --------------------------------------------------------------------------------
    metrics = pd.DataFrame(rows)
    move = pd.DataFrame(moves)
    var = pd.DataFrame(variants)
    comp = pd.DataFrame(comps)
    sbs = side_by_side(comp, float(config.alpha))
    # sign-flipped display rows (texture minus shape)
    flips = {k: v for f in fams.values() for k, v in (f.get("flipped") or {}).items()}
    counts_df, drivers_df = pd.DataFrame(counts), pd.DataFrame(drivers)
    metrics.to_csv(out / "metrics.csv", index=False)
    move.to_csv(out / "movement.csv", index=False)
    var.to_csv(out / "weighting_variants.csv", index=False)
    comp.to_csv(out / "comparisons.csv", index=False)
    sbs.to_csv(out / "comparisons_side_by_side.csv", index=False)
    counts_df.to_csv(out / "case_counts.csv", index=False)
    drivers_df.to_csv(out / "auroc_drivers.csv", index=False)
    np.savez_compressed(out / "replicates.npz", **replicate_store)
    checks["bootstrap"] = {
        "iCAIRD / PANDA Radboud": "slide resamples of the pooled versions, seed 20261108, B = 2000, stratified by "
        "slide label (positive / negative)",
        "MMCI": "patient resamples of the pooled MMCI row (tile-level run e97db5a5), seed 20261114, B = 2000",
        "weights": "per drawn case copy, 1 / n(case, class); duplicates are separate cases",
        "ci": f"{float(config.metrics_ci_level):.0%} percentile",
        "p": "2 min(P(d <= 0), P(d >= 0))",
        "holm": "per family and nuclei subset, over the AUPRC rows (as the pooled versions); AUROC: raw p",
    }
    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=str))
    (out / "report.md").write_text(
        report(
            metrics, move, var, comp, sbs, counts_df, drivers_df, flips, checks, config
        )
    )
    (out / "config-resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    return out, checks, private


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def ci(point: float, lo: float, hi: float, signed: bool = False) -> str:
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(point)} [{f.format(lo)}, {f.format(hi)}]"


def fmt_p(p: float) -> str:
    return "—" if pd.isna(p) else ("<0.001" if p < 0.001 else f"{p:.3f}")


def report(
    metrics, move, var, comp, sbs, counts, drivers, flips, checks, config
) -> str:
    def row(set_, sub, model, m, wt):
        return metrics[
            (metrics["set"] == set_)
            & (metrics["nuclei"] == sub)
            & (metrics["model"] == model)
            & (metrics["metric"] == m)
            & (metrics["weighting"] == wt)
        ].iloc[0]

    def mv(set_, sub, model, m):
        return move[
            (move["set"] == set_)
            & (move["nuclei"] == sub)
            & (move["model"] == model)
            & (move["metric"] == m)
        ].iloc[0]

    L = [
        "# Case-balanced evaluation: pooled vs per-case weights",
        "",
        (
            "Each case (slide; MMCI patient) contributes equally within its class: w = 1 / n(case, class). "
            "Seed means; 95 % percentile CIs from the pooled versions' resamples; change = case-balanced − pooled, "
            "paired per resample. Calibrated AUPRC: π₀ = 0.5 on the weighted classes (= TPR / (TPR + FPR) "
            "averaged over recall). The case-balanced AUPRC has the prevalence of the weighted classes "
            "(cases with carcinoma / all class-cases), not the nucleus prevalence."
        ),
        "",
        "## Pooled vs case-balanced",
        "",
        "| Set | Nuclei | Model | Metric | Pooled [95% CI] | Case-balanced [95% CI] | Change [95% CI] |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    items = [
        ("iCAIRD", "all", "shape", "Shape + RoVE"),
        ("iCAIRD", "all", "position", "Position only + RoVE"),
        ("PANDA Radboud", "all", config.radboud.main_model, "Main model"),
        ("PANDA Radboud", "epithelium", config.radboud.main_model, "Main model"),
        ("MMCI", "all", "mmci", "Main model (patient = case)"),
    ]
    for set_, sub, model, lab in items:
        for m in METRICS:
            a, b, d = (
                row(set_, sub, model, m, "pooled"),
                row(set_, sub, model, m, "balanced"),
                mv(set_, sub, model, m),
            )
            L.append(
                f"| {set_} | {sub} | {lab} | {NAMES[m]} | {ci(a.point, a.ci_lo, a.ci_hi)} | "
                f"{ci(b.point, b.ci_lo, b.ci_hi)} | {ci(d.change, d.change_ci_lo, d.change_ci_hi, True)} |"
            )
    L += ["", "Prevalence of the evaluated classes:", ""]
    for set_, sub, key in (
        ("iCAIRD", "all", ("iCAIRD",)),
        ("PANDA Radboud", "all", ("PANDA Radboud", "all")),
        ("PANDA Radboud", "epithelium", ("PANDA Radboud", "epithelium")),
        ("MMCI", "all", ("MMCI",)),
    ):
        c = checks["sets"]
        for k in key:
            c = c[k]
        c = c["counts"]
        L.append(
            f"- {set_} ({sub}): pooled {c['prevalence_pooled']:.3f} ({c['carcinoma_nuclei']:,} / "
            f"{c['nuclei']:,} nuclei); case-balanced {c['prevalence_balanced']:.3f} "
            f"({c['cases_with_carcinoma']} cases with carcinoma, {c['cases_with_benign']} with benign nuclei)"
        )

    L += [
        "",
        "## iCAIRD: shape − position only",
        "",
        "| Metric | Pooled Δ [95% CI] | p | Case-balanced Δ [95% CI] | p |",
        "| --- | --- | --- | --- | --- |",
    ]
    ic = sbs[sbs["set"] == "iCAIRD"]
    for m in METRICS:
        r = ic[ic["metric"] == m].iloc[0]
        L.append(
            f"| {NAMES[m]} | {ci(r.difference_pooled, r.ci_lo_pooled, r.ci_hi_pooled, True)} | "
            f"{fmt_p(r.p_pooled)} | {ci(r.difference_balanced, r.ci_lo_balanced, r.ci_hi_balanced, True)} | "
            f"{fmt_p(r.p_balanced)} |"
        )

    L += [
        "",
        "## Labelled nuclei per case",
        "",
        (
            "| Set | Nuclei | Class | Cases | Min | Median | Max | Largest case share | Largest 10 % of cases | "
            "Effective cases (pooled) |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in counts.itertuples():
        L.append(
            f"| {r.set} | {r.nuclei} | {r.nucleus_class} | {r.cases} | {r.min:,} | {r.median:,.0f} | {r.max:,} | "
            f"{r.largest_case_share:.3f} | {r.largest_10pct_cases_share:.3f} | {r.effective_cases_pooled:.1f} |"
        )

    L += [
        "",
        "## Where the change comes from: the four weightings (seed means)",
        "",
        "| Set | Nuclei | Model | Metric | Pooled | Carcinoma cases balanced | Benign cases balanced | Both |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in var.itertuples():
        if r.set == "PANDA Radboud" and r.model != config.radboud.main_model:
            continue
        # uncalibrated AP with one class per case and the other per nucleus: class totals not comparable
        mixed = "—" if r.metric == "AUPRC" else None
        L.append(
            f"| {r.set} | {r.nuclei} | {r.model} | {NAMES[r.metric]} | {r.pooled:.3f} | "
            f"{mixed or f'{r.pos_balanced:.3f}'} | {mixed or f'{r.neg_balanced:.3f}'} | {r.balanced:.3f} |"
        )
    L += [
        "",
        (
            "Mixed weightings leave the uncalibrated AUPRC undefined in practice (one class summed over cases, the "
            "other over nuclei), hence —. AUPRC at the pooled prevalence = the case-balanced curve evaluated at the "
            "nucleus prevalence: the AUPRC change without the change of prevalence."
        ),
        "",
        "AUROC of every PANDA Radboud configuration under the four weightings (seed means):",
        "",
        "| Nuclei | Configuration | Pooled | Carcinoma cases balanced | Benign cases balanced | Both |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in var[
        (var["set"] == "PANDA Radboud") & (var["metric"] == "AUROC")
    ].itertuples():
        L.append(
            f"| {r.nuclei} | {r.model} | {r.pooled:.3f} | {r.pos_balanced:.3f} | {r.neg_balanced:.3f} | "
            f"{r.balanced:.3f} |"
        )
    L += [
        "",
        (
            "ΔAUROC (case-balanced − pooled) by case size third within each class (seed means; step 1 "
            "balances the carcinoma cases against the pooled benign nuclei, step 2 then the benign cases):"
        ),
        "",
        (
            "| Set | Nuclei | Model | Step | Group | Cases | Nuclei per case | Pooled weight | Balanced weight | "
            "Mean score | Contribution |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in drivers.itertuples():
        L.append(
            f"| {r.set} | {r.nuclei} | {r.model} | {r.step} | {r.group} | {r.cases} | "
            f"{r.nuclei_min:,}–{r.nuclei_max:,} | {r.pooled_weight:.3f} | {r.balanced_weight:.3f} | "
            f"{r.mean_score:.3f} | {r.contribution_to_delta_auroc:+.4f} |"
        )
    L += [
        "",
        (
            "Score: carcinoma rows A = AUROC of the case's carcinoma nuclei vs the pooled benign nuclei; benign "
            "rows U = P(a benign nucleus of the case outranks a carcinoma nucleus), lower is better."
        ),
        "",
    ]

    L += [
        "## PANDA Radboud comparisons: pooled vs case-balanced",
        "",
        (
            "Holm per family and nuclei subset over the AUPRC rows (as the pooled versions); AUROC rows: raw p. "
            "Flag: SIGN = the point estimate changes sign; SIGNIFICANCE = significant at 0.05 (Holm for AUPRC, raw p "
            "for AUROC) in one weighting only."
        ),
        "",
        (
            "| Family | Nuclei | Comparison | Metric | Pooled Δ [95% CI] | p (Holm) | Case-balanced Δ [95% CI] | "
            "p (Holm) | Flag |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    rb = sbs[sbs["set"] == "PANDA Radboud"]
    for r in rb[rb["metric"].isin(["AUPRC", "AUROC"])].itertuples():
        if not r.report and r.comparison not in flips:
            continue
        s = -1 if r.comparison in flips else 1
        lab = flips.get(r.comparison, r.label)
        pp = fmt_p(r.p_holm_pooled if pd.notna(r.p_holm_pooled) else r.p_pooled)
        pb = fmt_p(r.p_holm_balanced if pd.notna(r.p_holm_balanced) else r.p_balanced)
        lo_p, hi_p = sorted((s * r.ci_lo_pooled, s * r.ci_hi_pooled))
        lo_b, hi_b = sorted((s * r.ci_lo_balanced, s * r.ci_hi_balanced))
        L.append(
            f"| {r.family} | {r.nuclei} | {lab} | {r.metric} | {ci(s * r.difference_pooled, lo_p, hi_p, True)} | "
            f"{pp} | {ci(s * r.difference_balanced, lo_b, hi_b, True)} | {pb} | {r.flag} |"
        )
    other = rb[
        (~rb["report"])
        & (~rb["comparison"].isin(list(flips)))
        & (rb["flag"] != "")
        & rb["metric"].isin(["AUPRC", "AUROC"])
    ]
    L += [
        "",
        "Other members of the Holm families with a flag: "
        + (
            "none."
            if other.empty
            else "; ".join(
                f"{r.family}/{r.nuclei} {r.label} ({r.metric}): {r.flag}"
                for r in other.itertuples()
            )
        ),
    ]
    calflags = rb[(rb["metric"] == "AUPRC_cal") & (rb["flag"] != "")]
    L += [
        "",
        "Calibrated AUPRC differences (not in the pooled families; raw p) with a flag: "
        + (
            "none."
            if calflags.empty
            else "; ".join(
                f"{r.family}/{r.nuclei} {r.label}: {r.flag}"
                for r in calflags.itertuples()
            )
        ),
    ]

    L += ["", "## Checks", ""]
    for name, c in checks["sets"].items():
        L.append(
            f"- {name}: `{json.dumps({k: v for k, v in c.items() if k != 'counts'}, default=str)[:1500]}`"
        )
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------


def preflight(out: Path, private_ids: list[str]) -> None:
    ids = re.compile(
        r"(?<![\w.,])("
        + "|".join(map(re.escape, sorted(set(private_ids))))
        + r")(?![\w.,])|TP-\d{4}_"
    )
    for path in out.iterdir():
        if path.is_dir():
            continue
        if path.suffix in {".csv", ".json", ".md", ".yaml"}:
            text = path.read_text()
        elif path.suffix == ".npz":
            z = np.load(path)
            text = (
                " ".join(z.files)
                + " "
                + " ".join(
                    " ".join(map(str, z[k].ravel()))
                    for k in z.files
                    if z[k].dtype.kind in "US"
                )
            )
        else:
            continue
        assert not CREDENTIAL_URL.search(text), f"credential in {path.name}"
        assert not SLIDE_ID.search(text), f"iCAIRD slide id in {path.name}"
        assert ids.search(text) is None, f"private id in {path.name}"


ARTIFACTS = (
    "metrics.csv",
    "movement.csv",
    "weighting_variants.csv",
    "comparisons.csv",
    "comparisons_side_by_side.csv",
    "case_counts.csv",
    "auroc_drivers.csv",
    "checks.json",
    "report.md",
    "replicates.npz",
)


@with_cli_args(["+postprocessing=case_balanced"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
def main(config: DictConfig) -> None:
    out, _, private = compute(config)
    preflight(out, private)
    if not config.log_mlflow:
        print(f"log_mlflow=false: outputs in {out}")
        return
    set_tracking(config.mlflow_tracking_uri)
    client = MlflowClient()
    metrics = pd.read_csv(out / "metrics.csv")
    move = pd.read_csv(out / "movement.csv")
    sbs = pd.read_csv(out / "comparisons_side_by_side.csv")
    tags = {
        "mlflow.runName": config.metadata.run_name,
        "mlflow.note.content": config.metadata.description,
        **git_info(*CODE),
    }
    params = {
        "pi0_reference": config.pi0_reference,
        "weights": "1 / n(case, class)",
        "resamples_icaird_radboud": "seed 20261108, B 2000, stratified by slide label",
        "resamples_mmci": f"{config.mmci.resamples_run[:8]} seed {config.mmci.resamples_seed} (patients)",
        "icaird_reference_run": "b973c17b",
        "radboud_reference_runs": "4081f97a, 027eeaee, 44ffd165",
        "mmci_reference_run": "e8682243",
    }
    for text in [*map(str, tags.values()), *map(str, params.values())]:
        assert not CREDENTIAL_URL.search(text), "credential URL in a tag or param"
    run_id = client.create_run(
        str(config.metadata.experiment_id), run_name=config.metadata.run_name, tags=tags
    ).info.run_id
    for k, v in params.items():
        client.log_param(run_id, k, v)
    key = {"iCAIRD": "icaird", "PANDA Radboud": "radboud", "MMCI": "mmci"}
    for r in metrics.itertuples():
        for suffix in ("point", "ci_lo", "ci_hi"):
            client.log_metric(
                run_id,
                f"{key[r.set]}/{r.nuclei}/{r.model}/{r.weighting}/{r.metric}_{suffix}",
                float(getattr(r, suffix)),
            )
    for r in move.itertuples():
        if r.set == "PANDA Radboud" and r.model != config.radboud.main_model:
            continue
        client.log_metric(
            run_id,
            f"{key[r.set]}/{r.nuclei}/{r.model}/change/{r.metric}",
            float(r.change),
        )
    for r in sbs.itertuples():
        for wt in WEIGHTINGS:
            client.log_metric(
                run_id,
                f"{key[r.set]}/{r.nuclei}/{r.family}/{r.comparison}/{wt}/{r.metric}_diff",
                float(getattr(r, f"difference_{wt}")),
            )
    client.log_metric(
        run_id,
        "flags/radboud_comparisons_flagged",
        float(
            (
                (sbs["set"] == "PANDA Radboud")
                & sbs["flag"].fillna("").ne("")
                & sbs["metric"].isin(["AUPRC", "AUROC"])
            ).sum()
        ),
    )
    for name in ARTIFACTS:
        client.log_artifact(run_id, str(out / name))
    if (out / "note.md").exists():
        client.log_artifact(run_id, str(out / "note.md"))
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
        f"MLflow run {run_id} (experiment {config.metadata.experiment_id}); outputs in {out}"
    )


if __name__ == "__main__":
    main()
