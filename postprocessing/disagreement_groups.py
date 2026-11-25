# ruff: noqa: RUF001, RUF002, RUF003, ISC004  (mathematical symbols in text are intended)
r"""Recall and specificity differences between configurations in model-independent groups.

PANDA Radboud test set, all 14.7 M nuclei; stored test predictions (seed mean of three seeds) of
the with-context models of tab:res-sources (Q4: position only, shape, texture; k-NN + RoVE) and
of the context models of tab:res-context (Q3: self-only, k-NN, dense; no positions in attention).
Replaces the dominant-concept columns of `postprocessing.disagreement_net` (run cdaf1247).

Operating point. Each configuration is classified at its OWN Youden threshold: the score t
maximising TPR − FPR of p ≥ t (highest t on ties) for the seed-mean probability of every
validation nucleus (392 slides). Validation predictions were made with each evaluation run's code
(see the config); per seed, the Youden threshold is compared with the evaluation run's logged
`thresholds/val_nuclei_j`, and the seed-mean thresholds with those of focus_size_analysis_radboud.

Groups (one true class each):
- carcinoma nuclei by Gleason pattern G3 / G4 / G5: label-mask value at the centroid, vertex
  majority (ties → higher) if the centroid lies outside G3-5 (cache of run fa36b112);
- carcinoma nuclei by the share of carcinoma nuclei in their slide: < 5 %, 5–20 %, 20–50 %,
  > 50 % (lower edge inclusive; 50 % exactly falls in the last bin);
- benign nuclei by compartment: benign epithelium and stroma (vertex rule on the PANDA label
  masks, ≥ 50 % of the vertices; class cache of xai.epithelium_separation, as tab:res-context);
  benign nuclei in neither (background, mixed) are counted in checks.json only;
- benign nuclei on slides with at least one carcinoma nucleus, by the centroid distance to the
  nearest carcinoma nucleus: ≤ 50 µm, > 50 µm (level-0 centroids, 0.4862 µm/px);
- overall: every carcinoma nucleus, every benign nucleus.

Per group and pair X vs Y: Δ = share(X correct, Y wrong) − share(Y correct, X wrong) = metric_X −
metric_Y in percentage points; metric = recall (carcinoma groups) or specificity (benign groups).
95 % percentile CIs from the shared stratified slide bootstrap (seed 20261108, B = 2,000; saved
resamples, not redrawn): in each resample the per-slide counts are weighted by the slide's
multiplicity. p = 2 × the smaller tail fraction of Δ, unadjusted (descriptive). The per-model
recall / specificity per group get CIs the same way.

Outputs in `out_dir` (no slide ids): `thresholds.csv`, `groups.csv` (sizes), `metrics.csv`
(per-model recall / specificity), `deltas.csv` (full precision), `groups_table.csv` (rounded, one
row per pair and group: the table to paste), `report.md`, `checks.json`,
`figures/res_groups_{sources,context}.{pdf,png}`. `cache/` (per-nucleus distance; no slide ids)
is not logged.
"""

import json
import multiprocessing as mp
import subprocess
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from scipy.spatial import cKDTree
from tqdm import tqdm

from nuclei_graph.mlflow_utils import tag_parent_run
from postprocessing.bootstrap_sources import (
    bootstrap_p,
    label_predictions,
    load_labels,
    load_run,
    nucleus_keys,
)
from postprocessing.disagreement_analysis import configurations, test_slides
from postprocessing.patient_score_quantiles_mmci import youden_threshold
from postprocessing.results_context import REPO, epithelium_classes
from visualization.disagreement_groups_figure import save_group_figures


SOURCES = (
    "postprocessing/disagreement_groups.py",
    "visualization/disagreement_groups_figure.py",
    "configs/postprocessing/disagreement_groups.yaml",
    "postprocessing/disagreement_analysis.py",
    "postprocessing/bootstrap_sources.py",
    "postprocessing/results_context.py",
    "configs/postprocessing/results_context.yaml",
)
STROMA, BENIGN_EPI = 1, 2
METRIC = {"carcinoma": "recall", "benign": "specificity"}


def code_commit() -> str:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    dirty = git("status", "--porcelain", "--untracked-files=no", "--", *SOURCES)
    return f"{git('rev-parse', 'HEAD')}{' (dirty)' if dirty else ''}"


# --------------------------------------------------------------------------
# Thresholds (validation)
# --------------------------------------------------------------------------


def logged_thresholds(runs: list[str]) -> list[float | None]:
    """thresholds/val_nuclei_j of the evaluation runs (None without MLflow access)."""
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        return [
            client.get_run(r).data.metrics.get("thresholds/val_nuclei_j") for r in runs
        ]
    except Exception as e:  # noqa: BLE001  (descriptive check only)
        print(f"logged thresholds unavailable: {e}")
        return [None] * len(runs)


def validation_files(config: DictConfig, name: str, runs: list[str]) -> list[Path]:
    v = config.validation
    if v.source[name] == "probes":
        return [Path(v.probes_dir) / f"{r}_validation.parquet" for r in runs]
    files = []
    for r, (ev, commit, training) in zip(runs, v.inference_files[name], strict=True):
        assert r.startswith(ev), f"{name}: inference file of {ev}, run {r}"
        files.append(Path(v.inference_dir) / commit / f"{training}_validation.parquet")
    return files


def thresholds(
    config: DictConfig, runs: dict[str, list[str]]
) -> tuple[pd.DataFrame, dict]:
    v = config.validation
    slides = pd.read_parquet(v.slides).sort_values("slide_id", ignore_index=True)
    labels = load_labels(slides, Path(config.supervision_dir), config.label_column)
    reference = json.loads(Path(v.reference_thresholds).read_text())
    ref_keys, y, missing = None, None, None
    rows = []
    for name, eval_runs in runs.items():
        probs = []
        for f in validation_files(config, name, eval_runs):
            d = pd.read_parquet(f)
            if ref_keys is None:
                ref_keys = d[["slide", "key"]].copy()
                y, missing = label_predictions(
                    ref_keys, labels, slides["is_carcinoma"].to_numpy(bool)
                )
            assert np.array_equal(
                d["slide"].to_numpy(), ref_keys["slide"].to_numpy()
            ), f
            assert np.array_equal(d["key"].to_numpy(), ref_keys["key"].to_numpy()), f
            probs.append(d["p"].to_numpy(np.float64))
        mean = np.mean(probs, axis=0)
        t, j = youden_threshold(y, mean)
        pred = mean >= t
        seeds = [youden_threshold(y, p)[0] for p in probs]
        logged = logged_thresholds(eval_runs)
        ref_name = v.reference_names.get(name)
        ref_t = reference[ref_name]["threshold"] if ref_name in reference else None
        if ref_t is not None:
            assert abs(ref_t - t) < 1e-4, f"{name}: Youden {t} != reference {ref_t}"
        rows.append(
            {
                "configuration": name,
                "label": config.q4_configurations[name].label
                if name in config.q4_configurations
                else config.configurations[name].label,
                "threshold": t,
                "youden_J": j,
                "validation_recall": float(pred[y == 1].mean()),
                "validation_specificity": float((~pred[y == 0]).mean()),
                **{f"seed{i}_threshold": s for i, s in enumerate(seeds, 1)},
                **{
                    f"seed{i}_logged_val_nuclei_j": lg for i, lg in enumerate(logged, 1)
                },
                "focus_size_analysis_threshold": ref_t,
                "validation_source": v.source[name],
            }
        )
        print(f"{name}: Youden {t:.4f} (J {j:.4f})", flush=True)
    assert y is not None
    info = {
        "validation_slides": len(slides),
        "validation_nuclei": len(y),
        "validation_carcinoma_nuclei": int(y.sum()),
        "validation_carcinoma_slide_nuclei_without_label_row": missing,
    }
    return pd.DataFrame(rows), info


# --------------------------------------------------------------------------
# Test inputs and groups
# --------------------------------------------------------------------------


def remap(path: str, config: DictConfig) -> str:
    remote, local = str(config.remote_project_path), str(config.inputs_path)
    return local + path[len(remote) :] if path.startswith(remote) else path


def slide_nearest(job: dict) -> NDArray[np.float32]:
    """Centroid distance (µm) of every nucleus of a slide to its nearest carcinoma nucleus.

    Nuclei in id order (= key order); carcinoma nuclei get 0 (themselves).
    """
    d = pd.read_parquet(job["path"], columns=["id", "centroid"]).sort_values(
        "id", ignore_index=True
    )
    keys = nucleus_keys(d["id"])
    assert np.array_equal(keys, np.asarray(job["keys"], np.uint64)), "nuclei differ"
    xy = np.stack(d["centroid"].to_numpy()).astype(np.float64)
    y = np.asarray(job["y"], bool)
    dist, _ = cKDTree(xy[y]).query(xy, k=1, workers=1)
    return (dist * float(job["mpp"])).astype(np.float32)


def nearest_carcinoma(
    config: DictConfig, slides: pd.DataFrame, slide: NDArray, key: NDArray, y: NDArray
) -> NDArray[np.float32]:
    """Distance to the nearest carcinoma nucleus; NaN on slides without carcinoma."""
    cache = Path(config.out_dir) / "cache" / "nearest_carcinoma_um.npy"
    if cache.exists():
        out = np.load(cache)
        assert len(out) == len(y)
        return out
    xai = pd.read_parquet(config.epithelium.slides).set_index("slide_id")
    starts = np.r_[0, np.cumsum(np.bincount(slide, minlength=len(slides)))]
    jobs, which = [], []
    for s, sid in enumerate(slides["slide_id"]):
        lo, hi = starts[s], starts[s + 1]
        if not y[lo:hi].any():
            continue
        which.append(s)
        jobs.append(
            {
                "path": remap(xai.loc[sid, "slide_nuclei_path"], config),
                "mpp": xai.loc[sid, "mpp_x"],
                "keys": key[lo:hi],
                "y": y[lo:hi] == 1,
            }
        )
    out = np.full(len(y), np.nan, np.float32)
    with mp.get_context("spawn").Pool(int(config.groups.workers)) as pool:
        for s, dist in zip(
            which,
            tqdm(
                pool.imap(slide_nearest, jobs, chunksize=4),
                total=len(jobs),
                desc="distance",
            ),
            strict=True,
        ):
            out[starts[s] : starts[s + 1]] = dist
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


def gleason_patterns(
    config: DictConfig, slides: pd.DataFrame, slide: NDArray, key: NDArray, y: NDArray
) -> tuple[NDArray[np.int8], dict]:
    """Pattern 3-5 of every carcinoma nucleus (0 otherwise), as run fa36b112."""
    pattern = np.zeros(len(y), np.int8)
    starts = np.r_[0, np.cumsum(np.bincount(slide, minlength=len(slides)))]
    outside = 0
    for s, sid in enumerate(slides["slide_id"]):
        lo, hi = starts[s], starts[s + 1]
        if not y[lo:hi].any():
            continue
        d = np.load(Path(config.gleason.cache_dir) / f"{sid}.npz")
        assert np.array_equal(d["key"], key[lo:hi]), "pattern cache: nuclei differ"
        assert np.array_equal(d["y_rule"], y[lo:hi]), (
            "pattern cache: vertex rule != label"
        )
        centroid, vertex = (
            d["centroid"].astype(np.int8),
            d["vertex_pattern"].astype(np.int8),
        )
        carc = y[lo:hi] == 1
        inside = carc & (centroid >= 3)
        p = np.zeros(hi - lo, np.int8)
        p[inside] = centroid[inside]
        p[carc & ~inside] = vertex[carc & ~inside]
        outside += int((carc & ~inside).sum())
        pattern[lo:hi] = p
    assert (pattern[y == 1] >= 3).all() and not pattern[y == 0].any()
    ref = json.loads(Path(config.gleason.checks).read_text())["test"]
    counts = {f"G{p}": int((pattern == p).sum()) for p in (3, 4, 5)}
    assert counts == ref["pattern_counts"], (counts, ref["pattern_counts"])
    assert outside == ref["carcinoma_centroid_outside_G3_5"]
    return pattern, {
        "pattern_counts": counts,
        "carcinoma_centroid_outside_G3_5": outside,
        "identical_to": config.gleason.run,
    }


def load_test(config: DictConfig, runs: dict[str, list[str]]) -> dict:
    slides = test_slides(config)
    slide_ids = slides["slide_id"].tolist()
    labels = load_labels(slides, Path(config.supervision_dir), config.label_column)
    reference, y, missing, p_mean = None, None, None, {}
    for name, eval_runs in runs.items():
        total = None
        for run_id in eval_runs:
            df, _ = load_run(
                run_id, slide_ids, Path(config.cache_dir), config.predictions_artifact
            )
            if reference is None:
                reference = df[["slide", "key"]].copy()
                y, missing = label_predictions(
                    df, labels, slides["is_carcinoma"].to_numpy(bool)
                )
            assert np.array_equal(df["slide"].to_numpy(), reference["slide"].to_numpy())
            assert np.array_equal(df["key"].to_numpy(), reference["key"].to_numpy())
            p = df["p"].to_numpy(np.float64)
            total = p if total is None else total + p
        p_mean[name] = total / len(eval_runs)
        print(f"{name}: test loaded", flush=True)
    assert reference is not None and y is not None
    slide = reference["slide"].to_numpy().astype(np.int64)
    key = reference["key"].to_numpy()
    return {
        "slides": slides,
        "slide": slide,
        "key": key,
        "y": y,
        "p_mean": p_mean,
        "labels_missing": missing,
    }


def build_groups(config: DictConfig, data: dict) -> tuple[list[dict], dict]:
    """(section, group, class, mask) in table order, and the checks."""
    slides, slide, key, y = data["slides"], data["slide"], data["key"], data["y"]
    n_slides = len(slides)
    carc, ben = y == 1, y == 0

    pattern, pattern_checks = gleason_patterns(config, slides, slide, key, y)
    cls, _ = epithelium_classes(config, slides, slide, y)

    per_slide_n = np.bincount(slide, minlength=n_slides)
    per_slide_c = np.bincount(slide, weights=carc, minlength=n_slides)
    share = per_slide_c / per_slide_n
    edges = np.asarray(config.groups.share_edges, float)
    share_bin = np.clip(
        np.searchsorted(edges, share, side="right") - 1, 0, len(edges) - 2
    )
    nucleus_bin = share_bin[slide]

    dist = nearest_carcinoma(config, slides, slide, key, y)
    on_carc_slide = (per_slide_c > 0)[slide]
    assert (
        np.isnan(dist[~on_carc_slide]).all() and not np.isnan(dist[on_carc_slide]).any()
    )
    near = float(config.groups.near_um)

    groups = []
    for p in (3, 4, 5):
        groups.append(("Gleason pattern", f"G{p}", "carcinoma", pattern == p))
    for b, label in enumerate(config.groups.share_labels):
        groups.append(
            ("carcinoma share of slide", label, "carcinoma", carc & (nucleus_bin == b))
        )
    groups.append(
        ("benign compartment", "benign epithelium", "benign", ben & (cls == BENIGN_EPI))
    )
    groups.append(("benign compartment", "stroma", "benign", ben & (cls == STROMA)))
    groups.append(
        (
            "distance to carcinoma",
            f"≤ {near:g} µm",
            "benign",
            ben & on_carc_slide & (dist <= near),
        )
    )
    groups.append(
        (
            "distance to carcinoma",
            f"> {near:g} µm",
            "benign",
            ben & on_carc_slide & (dist > near),
        )
    )
    groups.append(("overall", "all carcinoma", "carcinoma", carc))
    groups.append(("overall", "all benign", "benign", ben))
    out = [
        {"section": s, "group": g, "true_class": c, "mask": m} for s, g, c, m in groups
    ]

    slides_by_bin = {
        label: int(((share_bin == b) & (per_slide_c > 0)).sum())
        for b, label in enumerate(config.groups.share_labels)
    }
    checks = {
        "gleason": pattern_checks,
        "benign_by_class": {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(cls[ben], return_counts=True), strict=True)
        },
        "benign_neither_epithelium_nor_stroma": int(
            (ben & (cls != BENIGN_EPI) & (cls != STROMA)).sum()
        ),
        "slides_with_carcinoma": int((per_slide_c > 0).sum()),
        "carcinoma_slides_by_share_bin": slides_by_bin,
        "benign_nuclei_on_slides_with_carcinoma": int((ben & on_carc_slide).sum()),
        "benign_nuclei_on_slides_without_carcinoma": int((ben & ~on_carc_slide).sum()),
        "benign_distance_um_quantiles_5_25_50_75_95": [
            float(q)
            for q in np.quantile(
                dist[ben & on_carc_slide], [0.05, 0.25, 0.5, 0.75, 0.95]
            )
        ],
        "share_bin_rule": "lower edge inclusive; share = carcinoma nuclei / nuclei of the slide",
    }
    return out, checks


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


class Bootstrap:
    def __init__(self, config: DictConfig, slide: NDArray, n_slides: int) -> None:
        saved = np.load(config.resamples)
        self.w = saved["multiplicities"].astype(np.float64)
        assert self.w.shape[1] == n_slides
        self.slide, self.n_slides = slide, n_slides
        self.pct = (1 - config.bootstrap.ci_level) / 2 * 100

    def counts(self, mask: NDArray) -> NDArray:
        return np.bincount(self.slide[mask], minlength=self.n_slides).astype(np.float64)

    def ratio(self, num: NDArray, den: NDArray) -> NDArray:
        with np.errstate(invalid="ignore", divide="ignore"):
            return (self.w @ num) / (self.w @ den)

    def ci(self, reps: NDArray) -> tuple[float, float]:
        ok = np.isfinite(reps)
        lo, hi = np.percentile(reps[ok], [self.pct, 100 - self.pct])
        return float(lo), float(hi)


def tables(
    config: DictConfig, data: dict, groups: list[dict], th: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = data["y"]
    boot = Bootstrap(config, data["slide"], len(data["slides"]))
    t = th.set_index("configuration")["threshold"]
    correct = {n: (p >= t[n]) == (y == 1) for n, p in data["p_mean"].items()}

    size_rows, metric_rows, delta_rows = [], [], []
    for g in groups:
        mask = g["mask"]
        n_per = boot.counts(mask)
        key = {
            "section": g["section"],
            "group": g["group"],
            "true_class": g["true_class"],
            "metric": METRIC[g["true_class"]],
        }
        size_rows.append(
            key | {"n": int(n_per.sum()), "n_slides": int((n_per > 0).sum())}
        )
        for name, c in correct.items():
            k_per = boot.counts(mask & c)
            lo, hi = boot.ci(100 * boot.ratio(k_per, n_per))
            metric_rows.append(
                key
                | {
                    "configuration": name,
                    "threshold": float(t[name]),
                    "value_pct": 100 * k_per.sum() / n_per.sum(),
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
        for pair in config.pairs:
            cx, cy = correct[pair.x], correct[pair.y]
            nx_per, ny_per = boot.counts(mask & cx & ~cy), boot.counts(mask & cy & ~cx)
            reps = 100 * boot.ratio(nx_per - ny_per, n_per)
            ok = np.isfinite(reps)
            lo, hi = boot.ci(reps)
            n = n_per.sum()
            delta_rows.append(
                key
                | {
                    "pair": pair.name,
                    "question": pair.question,
                    "x": pair.x_short,
                    "y": pair.y_short,
                    "threshold_x": float(t[pair.x]),
                    "threshold_y": float(t[pair.y]),
                    "n": int(n),
                    "n_slides": int((n_per > 0).sum()),
                    "value_x_pct": 100 * (mask & cx).sum() / n,
                    "value_y_pct": 100 * (mask & cy).sum() / n,
                    "n_x_correct_y_wrong": int(nx_per.sum()),
                    "n_y_correct_x_wrong": int(ny_per.sum()),
                    "share_x_correct_y_wrong_pct": 100 * nx_per.sum() / n,
                    "share_y_correct_x_wrong_pct": 100 * ny_per.sum() / n,
                    "delta_pp": 100 * (nx_per.sum() - ny_per.sum()) / n,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "p": bootstrap_p(reps[ok]),
                    "resamples_undefined": int((~ok).sum()),
                }
            )
        print(f"{g['group']}: done", flush=True)
    deltas = pd.DataFrame(delta_rows)
    for r in deltas.itertuples():  # identity: Δ = metric_X − metric_Y
        assert abs(r.delta_pp - (r.value_x_pct - r.value_y_pct)) < 1e-9
    return pd.DataFrame(size_rows), pd.DataFrame(metric_rows), deltas


def paste_table(deltas: pd.DataFrame) -> pd.DataFrame:
    d = deltas
    return pd.DataFrame(
        {
            "question": d["question"],
            "comparison": d["x"] + " vs " + d["y"],
            "section": d["section"],
            "group": d["group"],
            "class": d["true_class"],
            "metric": d["metric"],
            "n_nuclei": d["n"],
            "n_slides": d["n_slides"],
            "threshold_X": d["threshold_x"].round(4),
            "threshold_Y": d["threshold_y"].round(4),
            "X_pct": d["value_x_pct"].round(2),
            "Y_pct": d["value_y_pct"].round(2),
            "X_correct_Y_wrong_pct": d["share_x_correct_y_wrong_pct"].round(2),
            "Y_correct_X_wrong_pct": d["share_y_correct_x_wrong_pct"].round(2),
            "delta_pp": d["delta_pp"].round(2),
            "ci_lo_pp": d["ci_lo"].round(2),
            "ci_hi_pp": d["ci_hi"].round(2),
        }
    )


def markdown(
    config: DictConfig,
    th: pd.DataFrame,
    sizes: pd.DataFrame,
    metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    checks: dict,
) -> str:
    lines = [
        "# Recall / specificity differences in model-independent groups (PANDA Radboud test)",
        "",
        "Δ = share(X correct, Y wrong) − share(Y correct, X wrong) = metric_X − metric_Y, "
        "percentage points; positive = X better. Carcinoma groups: recall; benign groups: "
        "specificity. Each configuration at its own Youden threshold (seed-mean validation "
        "probability). 95 % CI: stratified slide bootstrap (B = 2,000, seed 20261108).",
        "",
        "## Thresholds (validation, 392 slides, "
        f"{checks['validation']['validation_nuclei']:,} nuclei)",
        "",
        "| configuration | Youden t | J | val. recall | val. specificity | seeds' own t | logged per seed |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for r in th.itertuples():
        own = ", ".join(f"{getattr(r, f'seed{i}_threshold'):.4f}" for i in (1, 2, 3))
        lg = ", ".join(
            "–"
            if (v := getattr(r, f"seed{i}_logged_val_nuclei_j")) is None
            or not np.isfinite(v)
            else f"{v:.4f}"
            for i in (1, 2, 3)
        )
        lines.append(
            f"| {r.label} | {r.threshold:.4f} | {r.youden_J:.4f} | "
            f"{r.validation_recall:.4f} | {r.validation_specificity:.4f} | {own} | {lg} |"
        )
    lines += [
        "",
        "## Group sizes",
        "",
        "| section | group | class | nuclei | slides |",
        "|---|---|---|---:|---:|",
    ]
    for r in sizes.itertuples():
        lines.append(
            f"| {r.section} | {r.group} | {r.true_class} | {r.n:,} | {r.n_slides} |"
        )
    lines.append("")
    for pair in config.pairs:
        d = deltas[deltas["pair"] == pair.name]
        tx, ty = d["threshold_x"].iloc[0], d["threshold_y"].iloc[0]
        lines += [
            f"## {pair.question}: {pair.x_short} (t = {tx:.3f}) vs {pair.y_short} "
            f"(t = {ty:.3f}); positive: {pair.x_short} better",
            "",
            f"| group | metric | n | {pair.x_short} % | {pair.y_short} % | "
            "X>Y % | Y>X % | Δ pp [95 % CI] |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
        for r in d.itertuples():
            lines.append(
                f"| {r.group} | {r.metric} | {r.n:,} | {r.value_x_pct:.2f} | {r.value_y_pct:.2f} | "
                f"{r.share_x_correct_y_wrong_pct:.2f} | {r.share_y_correct_x_wrong_pct:.2f} | "
                f"{r.delta_pp:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] |"
            )
        lines.append("")
    lines += [
        "## Per-model recall / specificity (%, 95 % CI)",
        "",
        "| group | metric | " + " | ".join(th["configuration"]) + " |",
        "|---|---|" + "---|" * len(th),
    ]
    for (_, grp), part in metrics.groupby(["section", "group"], sort=False):
        part = part.set_index("configuration")
        cells = [
            f"{part.loc[c, 'value_pct']:.1f} [{part.loc[c, 'ci_lo']:.1f}, "
            f"{part.loc[c, 'ci_hi']:.1f}]"
            for c in th["configuration"]
        ]
        lines.append(
            f"| {grp} | {part['metric'].iloc[0]} | " + " | ".join(cells) + " |"
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def compute(config: DictConfig) -> tuple[Path, dict]:
    out = Path(config.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    runs = configurations(config)
    runs.pop("thesis", None)
    for pair in config.pairs:
        assert pair.x in runs and pair.y in runs, pair.name
    th, val_info = thresholds(config, runs)
    th.to_csv(out / "thresholds.csv", index=False)

    data = load_test(config, runs)
    groups, group_checks = build_groups(config, data)
    sizes, metrics, deltas = tables(config, data, groups, th)
    sizes.to_csv(out / "groups.csv", index=False)
    metrics.to_csv(out / "metrics.csv", index=False)
    deltas.to_csv(out / "deltas.csv", index=False)
    paste_table(deltas).to_csv(out / "groups_table.csv", index=False)

    y = data["y"]
    checks = {
        "nuclei": len(y),
        "slides": len(data["slides"]),
        "carcinoma_nuclei": int(y.sum()),
        "carcinoma_slide_nuclei_without_label_row": data["labels_missing"],
        "validation": val_info,
        "groups": group_checks,
        "configurations": runs,
        "resamples": str(config.resamples),
        "test_operating_point": {
            r.configuration: {
                "threshold": r.threshold,
                "recall": float(
                    (data["p_mean"][r.configuration][y == 1] >= r.threshold).mean()
                ),
                "specificity": float(
                    (data["p_mean"][r.configuration][y == 0] < r.threshold).mean()
                ),
            }
            for r in th.itertuples()
        },
    }
    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=str))
    (out / "report.md").write_text(markdown(config, th, sizes, metrics, deltas, checks))

    dirs = [out / "figures"]
    if config.get("repo_figures_dir"):
        dirs.append(Path(config.repo_figures_dir))
    save_group_figures(out, dirs, OmegaConf.to_container(config.figures))
    return out, checks


def metric_slug(text: str) -> str:
    """MLflow-safe metric name part: "≤ 50 µm" -> "le_50_um", "5–20 %" -> "5-20_pct"."""
    for a, b in (
        ("≤", "le "),
        ("<", "lt "),
        (">", "gt "),
        ("µ", "u"),
        ("–", "-"),
        ("%", "pct"),
    ):
        text = text.replace(a, b)
    return "_".join(text.split())


@autolog
def logged(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)
    client = logger.experiment
    client.set_tag(logger.run_id, "code_commit", code_commit())
    for key, value in config.links.items():
        client.set_tag(logger.run_id, key, value)
    out, _ = compute(config)

    metrics = {}
    for r in pd.read_csv(out / "thresholds.csv").itertuples():
        metrics[f"threshold/{r.configuration}"] = float(r.threshold)
    for r in pd.read_csv(out / "deltas.csv").itertuples():
        metrics[
            f"{r.pair}/{metric_slug(r.section)}/{metric_slug(r.group)}/delta_pp"
        ] = float(r.delta_pp)
    logger.log_metrics(metrics)
    for name in (
        "thresholds.csv",
        "groups.csv",
        "metrics.csv",
        "deltas.csv",
        "groups_table.csv",
        "report.md",
        "checks.json",
    ):
        logger.log_artifact(str(out / name))
    for stem in config.figures:
        for ext in ("pdf", "png"):
            logger.log_artifact(str(out / "figures" / f"{stem}.{ext}"), "figures")
    for source in SOURCES:
        logger.log_artifact(str(REPO / source), "code")
    (out / "config-resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    logger.log_artifact(str(out / "config-resolved.yaml"), "code")


@with_cli_args(["+postprocessing=disagreement_groups"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
def main(config: DictConfig) -> None:
    if config.compute_only:
        compute(config)
    else:
        logged(config)


if __name__ == "__main__":
    main()
