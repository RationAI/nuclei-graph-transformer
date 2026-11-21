# ruff: noqa: RUF001, RUF002  (mathematical symbols in text are intended)
r"""Gleason pattern and the SAE concepts of the thesis model; linear probes of h for G4+5 vs G3.

PANDA Radboud test set. Explained model c8f26853, SAE k = 16, L1, seed 1 (SAE training run
4cf1db8f), stable concepts A-H (deck key). Stored activations, codes and h only: no training of
the network, no inference.

Classes. A nucleus is carcinoma by the evaluation label (``annot_label``: ≥ 50 % of its polygon
vertices in the G3-5 mask; nuclei of ``is_carcinoma = False`` slides are 0; the XAI index label,
checked against the vertex rule recomputed here on every nucleus of the carcinoma slides). Its
Gleason pattern is the label-mask value (level 0, channel 0) at its centroid; a carcinoma nucleus
whose centroid lies outside the G3-5 mask takes the pattern with most of its vertices (ties →
higher; counted in ``checks.json``). Benign epithelium (reference row) = vertex-rule class 2 of
``xai.epithelium_separation`` (run 0b24e5db); stroma and benign nuclei are non-carcinoma.

(1) Test, per group G3 / G4 / G5 / G4+5 (carcinoma) and benign epithelium: nuclei, and per
    stable concept the share of active nuclei (κ > 0) and the mean κ. AUROC of each concept's κ
    for G4+5 (positive) vs G3 among the carcinoma nuclei (ties count half).
(2) Linear probes from the final representation h (output of ``final_norm``) to G4+5 vs G3 on
    carcinoma nuclei: L2 logistic regression on standardized inputs (mean / SD of the fit sample).
    Validation nuclei = the fixed 2,000,000-nucleus validation sample of tab:res-context-probes
    (seed 20261119), whose h of the self-only models was extracted there with each evaluation
    run's code. Its slides with carcinoma nuclei are split (seed ``probe.seed``, stratified by
    the slide's majority G4+5 vs G3) into fit and held-out slides (``heldout_slide_share``). For
    each C in ``C_grid`` a probe is fitted on a class-balanced sample of the fit slides (no slide
    above ``max_slide_share`` of it; per slide and class at most ⌊share · N⌋ nuclei, N per class
    maximal) and scored by AUROC on every carcinoma nucleus of the held-out slides; the smallest
    C whose held-out AUROC is within ``selection_tolerance`` of the best one is refitted on the
    balanced sample of all validation slides. Same samples for every model. Test: AUROC of the probe score on every carcinoma test nucleus. Self-only:
    the three seeds of tab:res-context (027eeaee), each with its own C; reported as the seed mean
    of the AUROC. Difference = analysed − self-only seed mean.

Intervals: 95 % percentile intervals from the shared stratified slide bootstrap (Q2 resamples,
seed 20261108, B = 2,000; read, not redrawn): in each resample a nucleus is weighted by its
slide's multiplicity. AUROCs per resample come from per-slide histograms on quantile bins of the
score, refined until the full-data histogram AUROC is within ``histogram.tolerance`` of the exact
one; shares and means are ratios of the weighted per-slide sums. p (difference) = 2 × the smaller
tail fraction.

Outputs in ``out_dir`` (no slide ids): ``groups.csv``, ``patterns.csv``, ``concept_auroc.csv``,
``probe_selection.csv``, ``probes.csv``, ``probe_summary.csv``, ``report.md``, ``checks.json``,
``replicates.npz``. The per-slide pattern cache (``cache_dir``, named by slide id) is not logged.
"""

import json
import multiprocessing as mp
import subprocess
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from openslide import OpenSlide
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from nuclei_graph.mlflow_utils import tag_parent_run
from nuclei_graph.xai.sae import encode, load_autoencoder
from postprocessing.bootstrap_sources import bootstrap_p, hist_metrics
from postprocessing.focus_geometry_radboud import (
    nucleus_keys,
    read_mask,
    vertex_classes,
)
from postprocessing.results_context import epithelium_classes
from xai.concept_morphometry import concept_letters


REPO = Path(__file__).parents[1]
SOURCES = (
    "postprocessing/gleason_concepts.py",
    "postprocessing/focus_geometry_radboud.py",
    "postprocessing/results_context.py",
    "postprocessing/bootstrap_sources.py",
    "configs/postprocessing/gleason_concepts.yaml",
)
ROWS = {"validation": "sample_validation.parquet", "test": "test_rows.parquet"}
H_TAG = {
    "validation": "validation",
    "test": "test_all",
}  # context_probes_extract outputs
GROUPS = ("G3", "G4", "G5", "G4+5", "benign epithelium")
BENIGN = 2


def code_commit() -> str:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    dirty = git("status", "--porcelain", "--untracked-files=no", "--", *SOURCES)
    return f"{git('rev-parse', 'HEAD')}{' (dirty)' if dirty else ''}"


# --------------------------------------------------------------------------
# Patterns (one worker per carcinoma slide)
# --------------------------------------------------------------------------


def slide_patterns(job: dict) -> str:
    """Caches key, centroid mask value, vertex rule and vertex-majority pattern per nucleus."""
    out = Path(job["out"])
    if out.exists():
        return str(out)
    nuclei = pd.read_parquet(
        job["nuclei_path"], columns=["id", "polygon", "centroid"]
    ).sort_values("id", ignore_index=True)
    mask = read_mask(Path(job["mask_path"]))
    my, mx = mask.shape
    with OpenSlide(job["slide_path"]) as slide:
        wx, wy = slide.dimensions
    sx, sy = mx / wx, my / wy
    y_rule, cls = vertex_classes(
        nuclei["polygon"].tolist(), mask, (sx, sy), float(job["overlap_threshold"])
    )
    cent = np.stack(nuclei["centroid"].to_numpy())  # (x, y), level 0
    cx = np.clip(np.floor(cent[:, 0] * sx).astype(int), 0, mx - 1)
    cy = np.clip(np.floor(cent[:, 1] * sy).astype(int), 0, my - 1)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        key=nucleus_keys(nuclei["id"]),
        centroid=mask[cy, cx].astype(np.uint8),
        y_rule=y_rule,
        vertex_pattern=np.where(y_rule == 1, cls, 0).astype(np.int8),
    )
    tmp.rename(out)
    return str(out)


def cache_patterns(
    config: DictConfig, split: str, slides: pd.DataFrame, which: NDArray
) -> None:
    jobs = [
        {
            "out": str(Path(config.cache_dir) / split / f"{row.slide_id}.npz"),
            "nuclei_path": row.slide_nuclei_path,
            "mask_path": str(
                Path(config.label_masks_dir) / f"{row.slide_id}_mask.tiff"
            ),
            "slide_path": row.slide_path,
            "overlap_threshold": float(config.overlap_threshold),
        }
        for row in slides.iloc[which].itertuples()
    ]
    todo = [j for j in jobs if not Path(j["out"]).exists()]
    missing = [j for j in todo if not Path(j["mask_path"]).exists()]
    assert not missing, f"{split}: {len(missing)} slides without a label mask"
    if todo:
        with mp.get_context("spawn").Pool(
            int(config.workers), maxtasksperchild=10
        ) as pool:
            list(
                tqdm(
                    pool.imap_unordered(slide_patterns, todo),
                    total=len(todo),
                    desc=split,
                )
            )


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def load_split(config: DictConfig, split: str) -> dict:
    """Rows (slide, key, xai_row), labels and Gleason pattern of a split's nuclei."""
    work = Path(config.probe_inputs_dir)
    slides = pd.read_parquet(work / f"slides_{split}.parquet")
    assert slides["slide_id"].is_monotonic_increasing and slides["slide_id"].is_unique
    rows = pd.read_parquet(work / ROWS[split])
    slide = rows["slide"].to_numpy().astype(np.int64)
    index = pd.read_parquet(config.xai.index[split], columns=["nucleus_id", "label"])
    xai_row = rows["xai_row"].to_numpy()
    assert np.array_equal(
        nucleus_keys(index["nucleus_id"].iloc[xai_row]), rows["key"].to_numpy()
    ), f"{split}: rows do not match the XAI index"
    y = index["label"].to_numpy()[xai_row].astype(np.int8)
    n_in_split = len(index)
    del index
    carcinoma_slide = slides["is_carcinoma"].to_numpy(bool)
    assert not y[~carcinoma_slide[slide]].any(), f"{split}: label 1 on a negative slide"

    which = np.flatnonzero(
        carcinoma_slide & (np.bincount(slide, minlength=len(slides)) > 0)
    )
    cache_patterns(config, split, slides, which)
    frames = []
    for s in which:
        d = np.load(Path(config.cache_dir) / split / f"{slides.slide_id.iloc[s]}.npz")
        frames.append(
            pd.DataFrame(
                {
                    "slide": np.full(len(d["key"]), s, np.int64),
                    "key": d["key"],
                    "centroid": d["centroid"],
                    "y_rule": d["y_rule"],
                    "vertex_pattern": d["vertex_pattern"],
                }
            )
        )
    cache = pd.concat(frames, ignore_index=True)
    merged = (
        rows[["slide", "key"]]
        .astype({"slide": np.int64})
        .merge(cache, on=["slide", "key"], how="left", validate="1:1")
    )
    on_carcinoma_slide = carcinoma_slide[slide]
    assert not merged["y_rule"].isna().to_numpy()[on_carcinoma_slide].any(), (
        f"{split}: nuclei of carcinoma slides without a cached pattern"
    )
    y_rule = merged["y_rule"].fillna(0).to_numpy(np.int8)
    mismatch = int((y_rule[on_carcinoma_slide] != y[on_carcinoma_slide]).sum())
    assert mismatch == 0, f"{split}: vertex rule != label on {mismatch} nuclei"

    centroid = merged["centroid"].fillna(0).to_numpy(np.int8)
    vertex = merged["vertex_pattern"].fillna(0).to_numpy(np.int8)
    carc = y == 1
    in_pattern = carc & (centroid >= 3)
    pattern = np.zeros(len(y), np.int8)
    pattern[in_pattern] = centroid[in_pattern]
    pattern[carc & ~in_pattern] = vertex[carc & ~in_pattern]
    assert (pattern[carc] >= 3).all() and not pattern[~carc].any()
    checks = {
        "nuclei": len(y),
        "nuclei_in_split": n_in_split,
        "slides": len(np.unique(slide)),
        "carcinoma_slides_cached": len(which),
        "carcinoma_nuclei": int(carc.sum()),
        "vertex_rule_vs_label_mismatch": mismatch,
        "carcinoma_centroid_outside_G3_5": int((carc & ~in_pattern).sum()),
        "carcinoma_centroid_mask_value_counts": {
            str(int(v)): int(c)
            for v, c in zip(*np.unique(centroid[carc], return_counts=True), strict=True)
        },
        "centroid_vs_vertex_pattern_agreement": float(
            (centroid[in_pattern] == vertex[in_pattern]).mean()
        ),
        "pattern_counts": {f"G{p}": int((pattern == p).sum()) for p in (3, 4, 5)},
    }
    return {
        "slides": slides,
        "slide": slide,
        "xai_row": xai_row,
        "y": y,
        "pattern": pattern,
        "checks": checks,
    }


def stable_order(config: DictConfig) -> tuple[list[int], list[str]]:
    letters = dict(concept_letters(config.xai.deck_key))
    stable = sorted(
        j for j, letter in letters.items() if letter in config.xai.stable_concepts
    )
    assert stable == sorted(config.xai.stable_indices), (
        stable,
        config.xai.stable_indices,
    )
    order = sorted(stable, key=lambda j: letters[j])
    return order, [letters[j] for j in order]


def read_rows(path: str, rows: NDArray) -> NDArray:
    """Rows of a (memory-mapped) .npy array, read in ascending order, returned in `rows` order."""
    a = np.load(path, mmap_mode="r")
    order = np.argsort(rows, kind="stable")
    out = np.empty((len(rows), *a.shape[1:]), a.dtype)
    step = 500_000
    for start in range(0, len(rows), step):
        part = order[start : start + step]
        out[part] = a[rows[part]]
    return out


def model_h(
    config: DictConfig, split: str, data: dict, sel: NDArray
) -> dict[str, list[NDArray]]:
    """H of the selected rows (positions in the split's row table) for every model and seed."""
    work = Path(config.probe_inputs_dir) / "outputs"
    out: dict[str, list[NDArray]] = {}
    for name, m in config.models.items():
        out[name] = []
        for run in m.runs:
            if name == "analysed":
                h = read_rows(config.xai.h[split], data["xai_row"][sel])
            else:
                path = work / f"{run}_{H_TAG[split]}_h.npy"
                assert np.load(path, mmap_mode="r").shape[0] == len(data["y"]), (
                    run,
                    split,
                )
                h = read_rows(str(path), sel)
            out[name].append(h)
            print(f"h {split} {name} {run[:8]}: {h.shape}", flush=True)
    return out


# --------------------------------------------------------------------------
# Bootstrap
# --------------------------------------------------------------------------


class Resamples:
    def __init__(
        self, config: DictConfig, slides: pd.DataFrame, slide: NDArray, y: NDArray
    ) -> None:
        saved = np.load(config.resamples)
        assert saved["slide_ids"].tolist() == slides["slide_id"].tolist(), (
            "resamples: slides"
        )
        positive = np.bincount(slide, weights=(y == 1), minlength=len(slides)) > 0
        assert np.array_equal(saved["positive"], positive), "resamples: strata"
        self.weights = saved["multiplicities"].astype(np.float64)
        self.seed = int(saved["seed"])
        self.n_slides = len(slides)
        self.pct = (1 - float(config.bootstrap.ci_level)) / 2 * 100

    def ci(self, reps: NDArray) -> tuple[float, float]:
        lo, hi = np.nanpercentile(reps, [self.pct, 100 - self.pct])
        return float(lo), float(hi)

    def per_slide(self, slide: NDArray, values: NDArray | None = None) -> NDArray:
        return np.bincount(slide, weights=values, minlength=self.n_slides)

    def auroc(
        self, score: NDArray, positive: NDArray, slide: NDArray, h: DictConfig
    ) -> dict:
        """Exact AUROC and its replicates (histogram on quantile bins, validated)."""
        exact = float(roc_auc_score(positive, score))
        n_bins = int(h.initial_bins)
        while True:
            edges = np.unique(np.quantile(score, np.linspace(0, 1, n_bins + 1)))
            bins = np.searchsorted(edges, score, side="left")
            nb = len(edges)
            full = hist_metrics(
                np.bincount(bins[positive], minlength=nb).astype(np.float64),
                np.bincount(bins[~positive], minlength=nb).astype(np.float64),
            )["AUROC"]
            err = float(full - exact)
            if abs(err) <= h.tolerance or n_bins >= h.max_bins:
                break
            n_bins *= 2
        assert abs(err) <= h.tolerance, f"histogram grid too coarse ({err:.2e})"
        flat = slide * nb + bins
        size = self.n_slides * nb
        pos = np.bincount(flat[positive], minlength=size).reshape(self.n_slides, nb)
        neg = np.bincount(flat[~positive], minlength=size).reshape(self.n_slides, nb)
        reps = []
        for start in range(0, len(self.weights), 250):
            w = self.weights[start : start + 250]
            reps.append(hist_metrics(w @ pos, w @ neg)["AUROC"])
        return {
            "exact": exact,
            "reps": np.concatenate(reps),
            "bins": nb,
            "hist_error": err,
        }


# --------------------------------------------------------------------------
# (1) Concepts per Gleason pattern
# --------------------------------------------------------------------------


def group_masks(test: dict, cls: NDArray) -> dict[str, NDArray]:
    p, y = test["pattern"], test["y"]
    return {
        "G3": p == 3,
        "G4": p == 4,
        "G5": p == 5,
        "G4+5": p >= 4,
        "benign epithelium": (cls == BENIGN) & (y == 0),
    }


def concept_tables(
    config: DictConfig,
    test: dict,
    cls: NDArray,
    kappa: NDArray,
    names: list[str],
    boot: Resamples,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    slide = test["slide"]
    masks = group_masks(test, cls)
    groups, rows = [], []
    w = boot.weights
    for g, m in masks.items():
        s = slide[m]
        k = kappa[m]
        count = boot.per_slide(s)
        rep_count = w @ count
        groups.append(
            {
                "group": g,
                "n_nuclei": int(m.sum()),
                "slides_with_nuclei": int((count > 0).sum()),
            }
        )
        for j, c in enumerate(names):
            active = k[:, j] > 0
            a_sum = boot.per_slide(s, active.astype(np.float64))
            k_sum = boot.per_slide(s, k[:, j].astype(np.float64))
            with np.errstate(invalid="ignore", divide="ignore"):
                share_reps = (w @ a_sum) / rep_count
                mean_reps = (w @ k_sum) / rep_count
            rows.append(
                {
                    "group": g,
                    "concept": c,
                    "n_nuclei": int(m.sum()),
                    "share_active": float(active.mean()),
                    "share_active_ci_lo": boot.ci(share_reps)[0],
                    "share_active_ci_hi": boot.ci(share_reps)[1],
                    "mean_kappa": float(k[:, j].astype(np.float64).mean()),
                    "mean_kappa_ci_lo": boot.ci(mean_reps)[0],
                    "mean_kappa_ci_hi": boot.ci(mean_reps)[1],
                    "mean_kappa_active": float(k[active, j].astype(np.float64).mean())
                    if active.any()
                    else np.nan,
                }
            )

    carc = test["y"] == 1
    positive = test["pattern"][carc] >= 4
    s_carc = slide[carc]
    auc_rows, reps, hist = [], {}, {}
    for j, c in enumerate(names):
        r = boot.auroc(
            kappa[carc, j].astype(np.float64), positive, s_carc, config.histogram
        )
        lo, hi = boot.ci(r["reps"])
        reps[f"concept_auroc/{c}"] = r["reps"]
        hist[c] = {"bins": r["bins"], "error": r["hist_error"]}
        auc_rows.append(
            {
                "concept": c,
                "auroc_g45_vs_g3": r["exact"],
                "ci_lo": lo,
                "ci_hi": hi,
                "n_g45": int(positive.sum()),
                "n_g3": int((~positive).sum()),
            }
        )
        print(f"concept {c}: AUROC {r['exact']:.3f} [{lo:.3f}, {hi:.3f}]", flush=True)
    return (
        pd.DataFrame(groups),
        pd.DataFrame(rows),
        pd.DataFrame(auc_rows),
        {"reps": reps, "histogram": hist},
    )


# --------------------------------------------------------------------------
# (2) Probes
# --------------------------------------------------------------------------


def balanced_sample(
    slide: NDArray, target: NDArray, max_share: float, rng: np.random.Generator
) -> NDArray:
    """Indices: N of each class, at most ⌊max_share · N⌋ per slide and class (≤ max_share of 2N)."""
    perm = rng.permutation(len(slide))
    s, t = slide[perm], target[perm]
    order = np.lexsort(
        (np.arange(len(perm)), t, s)
    )  # random order within (slide, class)
    cell = s[order] * 2 + t[order]
    start = np.r_[0, np.flatnonzero(np.diff(cell)) + 1]
    rank = np.arange(len(cell)) - np.repeat(start, np.diff(np.r_[start, len(cell)]))
    rank_perm = np.empty(len(cell), np.int64)
    rank_perm[order] = rank  # within-cell rank, in perm order
    sizes = {c: np.bincount(s[t == c]) for c in (0, 1)}  # nuclei per slide and class

    def capacity(n: int) -> int:
        cap = int(np.floor(max_share * n))
        return min(int(np.minimum(sizes[c], cap).sum()) for c in (0, 1))

    lo, hi = 0, int(min(sizes[0].sum(), sizes[1].sum()))
    while lo < hi:  # largest N with capacity(N) ≥ N
        mid = (lo + hi + 1) // 2
        if capacity(mid) >= mid:
            lo = mid
        else:
            hi = mid - 1
    n = lo
    cap = int(np.floor(max_share * n))
    chosen = []
    for c in (0, 1):
        pool = perm[(t == c) & (rank_perm < cap)]
        chosen.append(rng.choice(pool, n, replace=False))
    idx = np.sort(np.concatenate(chosen))
    share = np.bincount(slide[idx]).max() / len(idx)
    assert share <= max_share + 1e-12, share
    return idx


def probe_splits(config: DictConfig, val: dict) -> dict:
    """Fit / held-out slides and the balanced samples (positions among the carcinoma rows)."""
    p = config.probe
    rng = np.random.default_rng(int(p.seed))
    carc = np.flatnonzero(val["y"] == 1)
    slide = val["slide"][carc]
    target = (val["pattern"][carc] >= 4).astype(np.int64)
    slides = np.unique(slide)
    majority = (
        np.bincount(slide, weights=target)[slides] / np.bincount(slide)[slides] >= 0.5
    )
    heldout = []
    for stratum in (slides[majority], slides[~majority]):
        k = round(float(p.heldout_slide_share) * len(stratum))
        heldout.append(rng.permutation(stratum)[:k])
    heldout = np.sort(np.concatenate(heldout))
    is_held = np.isin(slide, heldout)
    fit_pos = np.flatnonzero(~is_held)
    fit = fit_pos[
        balanced_sample(slide[fit_pos], target[fit_pos], float(p.max_slide_share), rng)
    ]
    final = balanced_sample(slide, target, float(p.max_slide_share), rng)
    split_checks = {
        "validation_carcinoma_nuclei": len(carc),
        "validation_carcinoma_slides": len(slides),
        "heldout_slides": len(heldout),
        "heldout_nuclei": int(is_held.sum()),
        "heldout_g45_share": float(target[is_held].mean()),
        "fit_sample": len(fit),
        "fit_sample_slides": len(np.unique(slide[fit])),
        "fit_sample_max_slide_share": float(np.bincount(slide[fit]).max() / len(fit)),
        "final_sample": len(final),
        "final_sample_slides": len(np.unique(slide[final])),
        "final_sample_max_slide_share": float(
            np.bincount(slide[final]).max() / len(final)
        ),
        "final_sample_g45_share": float(target[final].mean()),
    }
    return {
        "carc": carc,
        "target": target,
        "fit": fit,
        "heldout": np.flatnonzero(is_held),
        "final": final,
        "checks": split_checks,
    }


class Probe:
    def __init__(self, x: NDArray, t: NDArray, c: float, max_iter: int) -> None:
        x64 = x.astype(np.float64)
        self.mu = x64.mean(0)
        sd = x64.std(0)
        self.sd = np.where(sd > 0, sd, 1.0)
        self.clf = LogisticRegression(C=c, max_iter=max_iter)
        self.clf.fit((x64 - self.mu) / self.sd, t)
        self.n_iter = int(self.clf.n_iter_[0])

    def score(self, x: NDArray, chunk: int = 1_000_000) -> NDArray:
        w = self.clf.coef_[0] / self.sd
        b = float(self.clf.intercept_[0] - self.mu @ w)
        out = np.empty(len(x))
        for start in range(0, len(x), chunk):
            out[start : start + chunk] = (
                x[start : start + chunk].astype(np.float64) @ w + b
            )
        return out


def probes(
    config: DictConfig, val: dict, test: dict, boot: Resamples
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict]:
    sp = probe_splits(config, val)
    h_val = model_h(config, "validation", val, sp["carc"])
    carc_test = np.flatnonzero(test["y"] == 1)
    positive = test["pattern"][carc_test] >= 4
    s_test = test["slide"][carc_test]
    h_test = model_h(config, "test", test, carc_test)
    t = sp["target"]
    max_iter = int(config.probe.max_iter)
    tol = float(config.probe.selection_tolerance)
    assert list(config.probe.C_grid) == sorted(config.probe.C_grid)

    sel_rows, rows, reps = [], [], {}
    for name, m in config.models.items():
        for seed, run in enumerate(m.runs, start=1):
            xv = h_val[name][seed - 1]
            curve = []
            for C in config.probe.C_grid:
                pr = Probe(xv[sp["fit"]], t[sp["fit"]], float(C), max_iter)
                auc = float(
                    roc_auc_score(t[sp["heldout"]], pr.score(xv[sp["heldout"]]))
                )
                curve.append((float(C), auc))
                sel_rows.append(
                    {
                        "model": name,
                        "seed": seed,
                        "run": run,
                        "C": float(C),
                        "heldout_auroc": auc,
                        "n_iter": pr.n_iter,
                    }
                )
                print(
                    f"{name} {seed} C={C:g}: held-out AUROC {auc:.4f} ({pr.n_iter} it)",
                    flush=True,
                )
            top = max(a for _, a in curve)
            best = next(
                (c, a) for c, a in curve if a >= top - tol
            )  # smallest C near the best
            final = Probe(xv[sp["final"]], t[sp["final"]], best[0], max_iter)
            score = final.score(h_test[name][seed - 1])
            r = boot.auroc(score, positive, s_test, config.histogram)
            lo, hi = boot.ci(r["reps"])
            reps[f"probe_auroc/{name}/{seed}"] = r["reps"]
            rows.append(
                {
                    "model": name,
                    "seed": seed,
                    "run": run,
                    "C": best[0],
                    "heldout_auroc": best[1],
                    "final_fit_auroc": float(
                        roc_auc_score(t[sp["final"]], final.score(xv[sp["final"]]))
                    ),
                    "n_iter": final.n_iter,
                    "test_auroc": r["exact"],
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "hist_bins": r["bins"],
                    "hist_error": r["hist_error"],
                }
            )
            print(
                f"{name} {seed}: C={best[0]:g}, test AUROC {r['exact']:.4f} [{lo:.4f}, {hi:.4f}]",
                flush=True,
            )
        del h_val[name], h_test[name]
    sel = pd.DataFrame(sel_rows)
    sel["selected"] = (
        sel.merge(
            pd.DataFrame(rows)[["model", "seed", "C"]].assign(selected=True),
            on=["model", "seed", "C"],
            how="left",
        )["selected"]
        .fillna(False)
        .to_numpy()
    )
    table = pd.DataFrame(rows)

    summary, summary_reps = [], {}
    for name in config.models:
        keys = [k for k in reps if k.startswith(f"probe_auroc/{name}/")]
        summary_reps[name] = np.mean([reps[k] for k in keys], axis=0)
        point = float(table.loc[table.model == name, "test_auroc"].mean())
        lo, hi = boot.ci(summary_reps[name])
        label = name if len(keys) == 1 else f"{name} (mean of {len(keys)} seeds)"
        summary.append(
            {"quantity": label, "auroc": point, "ci_lo": lo, "ci_hi": hi, "p": np.nan}
        )
    diff = summary_reps["analysed"] - summary_reps["self"]
    reps["probe_auroc_difference/analysed_minus_self"] = diff
    lo, hi = boot.ci(diff)
    summary.append(
        {
            "quantity": "analysed − self-only",
            "auroc": summary[0]["auroc"] - summary[1]["auroc"],
            "ci_lo": lo,
            "ci_hi": hi,
            "p": bootstrap_p(diff),
        }
    )
    checks = {
        **sp["checks"],
        "test_carcinoma_nuclei": len(carc_test),
        "test_g45_share": float(positive.mean()),
    }
    return sel, table, pd.DataFrame(summary), reps, checks


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def report(
    config: DictConfig,
    groups: pd.DataFrame,
    pats: pd.DataFrame,
    auc: pd.DataFrame,
    table: pd.DataFrame,
    summary: pd.DataFrame,
    names: list[str],
    checks: dict,
    boot: Resamples,
) -> str:
    def ci(v: float, lo: float, hi: float, f: str = ".3f") -> str:
        return f"{v:{f}} [{lo:{f}}, {hi:{f}}]"

    head = "| group | nuclei | " + " | ".join(names) + " |"
    sep = "|---|---:|" + "---:|" * len(names)
    lines = [
        "# Gleason pattern and the SAE concepts A–H (PANDA Radboud test)",
        "",
        (
            "Explained model c8f26853, SAE k = 16 L1 seed 1 (4cf1db8f). Carcinoma nuclei by the "
            "evaluation label; pattern = label-mask value at the centroid "
            f"({checks['test']['carcinoma_centroid_outside_G3_5']:,} carcinoma nuclei with the centroid "
            "outside G3–5 take their vertex-majority pattern). Benign epithelium = vertex-rule class 2. "
            f"Intervals: 95 % percentile, stratified slide bootstrap (B = {len(boot.weights):,}, seed {boot.seed})."
        ),
        "",
        "## (1) Share of active nuclei (κ > 0)",
        "",
        head,
        sep,
    ]
    n = groups.set_index("group")["n_nuclei"]
    for g in GROUPS:
        q = pats[pats.group == g].set_index("concept")
        lines.append(
            f"| {g} | {n[g]:,} | "
            + " | ".join(f"{q.loc[c, 'share_active']:.3f}" for c in names)
            + " |"
        )
    lines += [
        "",
        "## (1) Mean activation κ (all nuclei of the group, inactive = 0)",
        "",
        head,
        sep,
    ]
    for g in GROUPS:
        q = pats[pats.group == g].set_index("concept")
        lines.append(
            f"| {g} | {n[g]:,} | "
            + " | ".join(f"{q.loc[c, 'mean_kappa']:.3f}" for c in names)
            + " |"
        )
    a = auc.iloc[0]
    lines += [
        "",
        f"## (1) AUROC of κ for G4+5 (n = {a.n_g45:,}) vs G3 (n = {a.n_g3:,}), carcinoma nuclei",
        "",
        "> 0.5: higher activation in G4+5.",
        "",
        "| concept | AUROC [95 % CI] |",
        "|---|---|",
    ]
    for r in auc.itertuples():
        lines.append(f"| {r.concept} | {ci(r.auroc_g45_vs_g3, r.ci_lo, r.ci_hi)} |")
    pc = checks["probe"]
    lines += [
        "",
        "## (2) Linear probe h → G4+5 vs G3 (test carcinoma nuclei)",
        "",
        (
            "L2 logistic regression on standardized h; C chosen by AUROC on "
            f"{pc['heldout_slides']} held-out validation slides ({pc['heldout_nuclei']:,} carcinoma nuclei), "
            f"refitted on a balanced sample of {pc['final_sample']:,} validation nuclei from "
            f"{pc['final_sample_slides']} slides (max slide share {pc['final_sample_max_slide_share']:.4f}). "
            f"Test: {pc['test_carcinoma_nuclei']:,} carcinoma nuclei (G4+5 share {pc['test_g45_share']:.3f})."
        ),
        "",
        "| model | seed | C | held-out AUROC | test AUROC [95 % CI] |",
        "|---|---:|---:|---:|---|",
    ]
    for r in table.itertuples():
        lines.append(
            f"| {r.model} | {r.seed} | {r.C:g} | {r.heldout_auroc:.4f} | {ci(r.test_auroc, r.ci_lo, r.ci_hi, '.4f')} |"
        )
    lines += ["", "| quantity | AUROC [95 % CI] | p |", "|---|---|---|"]
    for r in summary.itertuples():
        p = (
            ""
            if np.isnan(r.p)
            else (f"< {2 / len(boot.weights):.4f}" if r.p == 0 else f"{r.p:.4f}")
        )
        lines.append(
            f"| {r.quantity} | {ci(r.auroc, r.ci_lo, r.ci_hi, '+.4f' if '−' in r.quantity else '.4f')} | {p} |"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def compute(config: DictConfig) -> tuple[Path, dict]:
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    order, names = stable_order(config)

    test = load_split(config, "test")
    val = load_split(config, "validation")
    rows = pd.read_parquet(
        Path(config.probe_inputs_dir) / ROWS["test"], columns=["slide", "key"]
    )
    assert (np.diff(rows["slide"].to_numpy()) >= 0).all()
    del rows
    boot = Resamples(config, test["slides"], test["slide"], test["y"])
    cls, cls_checks = epithelium_classes(
        config, test["slides"][["slide_id", "is_carcinoma"]], test["slide"], test["y"]
    )

    codes = read_rows(config.xai.codes_test, test["xai_row"])[:, order]
    # stored κ vs encoding the stored h, on carcinoma test rows
    sae, meta = load_autoencoder(config.xai.sae_model)
    assert (meta["k"], meta["seed"]) == (16, 1), meta
    carc = np.flatnonzero(test["y"] == 1)
    check = carc[
        np.linspace(0, len(carc) - 1, int(config.xai.code_check_rows)).astype(int)
    ]
    h_check = read_rows(config.xai.h.test, test["xai_row"][check])
    with torch.no_grad():
        enc = encode(sae, torch.as_tensor(h_check).float()).numpy()[:, order]
    code_dev = float(np.abs(enc - codes[check]).max())
    assert code_dev <= config.xai.code_tolerance, code_dev

    groups, pats, auc, part1 = concept_tables(config, test, cls, codes, names, boot)
    del codes
    sel, table, summary, probe_reps, probe_checks = probes(config, val, test, boot)

    checks = {
        "concepts": names,
        "test": test["checks"],
        "validation": val["checks"],
        "epithelium_label_vs_class": cls_checks,
        "stored_codes_vs_encoded_max_abs_diff": code_dev,
        "resamples": {
            "file": str(config.resamples),
            "n": len(boot.weights),
            "seed": boot.seed,
        },
        "concept_auroc_histogram": part1["histogram"],
        "probe": probe_checks,
    }
    groups.to_csv(out / "groups.csv", index=False)
    pats.to_csv(out / "patterns.csv", index=False)
    auc.to_csv(out / "concept_auroc.csv", index=False)
    sel.to_csv(out / "probe_selection.csv", index=False)
    table.to_csv(out / "probes.csv", index=False)
    summary.to_csv(out / "probe_summary.csv", index=False)
    np.savez_compressed(out / "replicates.npz", **part1["reps"], **probe_reps)  # type: ignore[arg-type]
    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=str))
    (out / "report.md").write_text(
        report(config, groups, pats, auc, table, summary, names, checks, boot)
    )
    return out, checks


OUTPUTS = (
    "groups.csv",
    "patterns.csv",
    "concept_auroc.csv",
    "probe_selection.csv",
    "probes.csv",
    "probe_summary.csv",
    "report.md",
    "checks.json",
    "replicates.npz",
)


@autolog
def logged(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)
    client = logger.experiment
    client.set_tag(logger.run_id, "code_commit", code_commit())
    for key, value in config.links.items():
        client.set_tag(logger.run_id, key, value)
    out, _ = compute(config)
    metrics = {}
    for r in pd.read_csv(out / "concept_auroc.csv").itertuples():
        metrics[f"concept_auroc/{r.concept}"] = r.auroc_g45_vs_g3
        metrics[f"concept_auroc/{r.concept}/ci_lo"] = r.ci_lo
        metrics[f"concept_auroc/{r.concept}/ci_hi"] = r.ci_hi
    for r in pd.read_csv(out / "probes.csv").itertuples():
        metrics[f"probe_auroc/{r.model}/seed{r.seed}"] = r.test_auroc
    s = pd.read_csv(out / "probe_summary.csv")
    for key, r in zip(
        ("analysed", "self_mean", "difference"), s.itertuples(), strict=True
    ):
        metrics[f"probe_auroc/{key}"] = r.auroc
        metrics[f"probe_auroc/{key}/ci_lo"] = r.ci_lo
        metrics[f"probe_auroc/{key}/ci_hi"] = r.ci_hi
    logger.log_metrics(metrics)
    for name in OUTPUTS:
        logger.log_artifact(str(out / name))
    for source in SOURCES:
        logger.log_artifact(str(REPO / source), "code")
    (out / "config-resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    logger.log_artifact(str(out / "config-resolved.yaml"), "code")


@with_cli_args(["+postprocessing=gleason_concepts"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
def main(config: DictConfig) -> None:
    if config.compute_only:
        compute(config)
    else:
        logged(config)


if __name__ == "__main__":
    main()
