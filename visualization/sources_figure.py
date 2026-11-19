"""fig:res-sources, redrawn from the saved outputs of the final information-source evaluation.

All test nuclei of the PANDA Radboud test set; the three with-context models (k-NN attention with
rotated values, k = 24, three seeds each): position only, shape and texture. Nothing is recomputed
from predictions: the inputs are `configurations.csv` and `replicates.npz` of
postprocessing.results_sources run 4081f97a (tab:res-sources, tab:res-positions); run 44ffd165
(results_sources_epithelium) reproduces the same all-nuclei rows and is checked against them.

One figure at the text width, two panels:
(a) AUPRC and (b) AUROC, one row per model: the seed-averaged point estimate (filled circle), its
    95 % percentile CI from the shared stratified slide bootstrap (B = 2000, seed 20261108; the
    seed mean is recomputed in every resample) and the three individual seeds (ticks above the
    bar). Value and CI are printed on the right of each panel.

The CIs are recomputed from the replicates and asserted equal to the stored ones, and the point
estimates are asserted to round to the values of tab:res-sources. With `--log`, the figure is
logged to MLflow experiment 37 as a child of the thesis model run (c8f26853); tracking URI and
credentials come from the environment.

Usage:
    uv run -m visualization.sources_figure [--log]
"""

import argparse
import json
import subprocess
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from matplotlib.transforms import blended_transform_factory

from visualization.thesis_results_figures import (
    FONT_SIZE,
    RC,
    TEXT_W_IN,
    save_figure,
)


REPO = Path(__file__).parents[1]
EVAL = Path("/mnt/projects/nuclei_based_wsi_analysis/evaluation/radboud_test")
RESULTS_DIR = EVAL / "q2_q3_positions"  # run 4081f97a
CHECK_DIR = EVAL / "q2_sources_epithelium"  # run 44ffd165, subset "all"
RESAMPLES = Path(
    "/mnt/public_projects/nuclei_based_wsi_analysis/evaluation/radboud_test/"
    "q2_sources_bootstrap/bootstrap_resamples_radboud_test.npz"
)
OUT_DIR = REPO / "figures"
STEM = "res_sources"

EXPERIMENT_ID = "37"
PARENT_RUN_ID = "c8f268536be8416f9fe395eea95c458e"
SOURCE_RUNS = {
    "results_run": "4081f97a7b3040a8a257599e1860d76c",
    "check_run": "44ffd16586d84e70967d5a4e5051594a",
}
N_RESAMPLES, RESAMPLE_SEED, CI_LEVEL = 2000, 20261108, 0.95

# top to bottom
MODELS = (
    ("position_knn", "positions", "#2E9A6B"),
    ("shape_knn", "positions + shape", "#3165AE"),
    ("texture_knn", "positions + texture", "#D6782A"),
)
EXPECTED = {  # tab:res-sources, three decimals
    "AUPRC": {"position_knn": 0.832, "shape_knn": 0.904, "texture_knn": 0.945},
    "AUROC": {"position_knn": 0.895, "shape_knn": 0.942, "texture_knn": 0.969},
}
PANELS = (
    # metric, x range, x ticks
    ("AUPRC", (0.80, 0.96), (0.80, 0.85, 0.90, 0.95)),
    ("AUROC", (0.87, 0.98), (0.88, 0.92, 0.96)),
)

INK, MUTED, FAINT, GRID = "#333333", "#7a7a7a", "#a0a0a0", "#e4e4e4"
FIG_H_IN = 2.25
BOTTOM, HEIGHT = 0.17, 0.66
AXES = ((0.17, 0.265), (0.59, 0.285))  # (left, width) of (a) and (b), figure fractions
VALUES_DX = 0.065  # centre of each numbers column, right of its axes
LETTER_DX = (-0.02, -0.03)  # panel letters, relative to the left of the labels / axes
LABELS_W = 0.14  # width of the row labels left of (a)
POINT_SIZE, CI_LW = 4.2, 2.2
# seed ticks: height above the bar (rows), marker size, line width
SEED_DY, SEED_TICK, SEED_MEW = 0.17, 4.5, 0.7


def bold(text: str) -> str:
    """Bold in Computer Modern (cmr10 has no bold face; mathtext's cm set has cmb10)."""
    return r"$\mathbf{" + text.replace(" ", r"\ ") + "}$"


def load_rows(results_dir: Path) -> pd.DataFrame:
    conf = pd.read_csv(results_dir / "configurations.csv")
    return conf[conf["configuration"].isin([m for m, _, _ in MODELS])].set_index(
        ["metric", "configuration"]
    )


def check(results_dir: Path, check_dir: Path, resamples: Path) -> dict:
    """Recomputes each CI from the bootstrap replicates and checks every input against the others."""
    rows = load_rows(results_dir)
    alpha = 100 * (1 - CI_LEVEL) / 2
    with np.load(resamples, allow_pickle=True) as z:
        assert int(z["seed"]) == RESAMPLE_SEED, int(z["seed"])
        assert z["multiplicities"].shape[0] == N_RESAMPLES, z["multiplicities"].shape
        n_slides = int(z["multiplicities"].shape[1])
    with np.load(results_dir / "replicates.npz", allow_pickle=True) as z:
        configs, eval_runs = z["configurations"], z["eval_runs"]
        reps = {m: z[m] for m, _, _ in PANELS}
    other = pd.read_csv(check_dir / "configurations.csv")
    other = other[other["subset"] == "all"].set_index(["metric", "configuration"])

    out: dict = {}
    for metric, _, _ in PANELS:
        assert reps[metric].shape[1] == N_RESAMPLES, reps[metric].shape
        for name, _, _ in MODELS:
            row = rows.loc[(metric, name)]
            runs = [row[f"run_{k}"] for k in (1, 2, 3)]
            sel = configs == name
            assert sel.sum() == 3 and sorted(eval_runs[sel]) == sorted(runs), name
            seed_mean = reps[metric][sel].mean(0)
            lo, hi = np.percentile(seed_mean, [alpha, 100 - alpha])
            assert np.isclose(lo, row["ci_lo"], atol=1e-9), (metric, name, lo)
            assert np.isclose(hi, row["ci_hi"], atol=1e-9), (metric, name, hi)
            seeds = [float(row[f"seed_{k}"]) for k in (1, 2, 3)]
            assert np.isclose(np.mean(seeds), row["point"], atol=1e-9), (metric, name)
            for key in ("point", "ci_lo", "ci_hi", "seed_1", "seed_2", "seed_3"):
                assert np.isclose(
                    other.loc[(metric, name), key], row[key], atol=1e-9
                ), (
                    "44ffd165 differs",
                    metric,
                    name,
                    key,
                )
            expected = EXPECTED[metric][name]
            if round(float(row["point"]), 3) != expected:
                raise SystemExit(
                    f"{metric} {name}: point {row['point']:.4f}, expected {expected:.3f}"
                )
            out[f"{metric}/{name}"] = {
                "point": float(row["point"]),
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "seeds": seeds,
                "eval_runs": runs,
            }
    out["bootstrap"] = {
        "n_resamples": N_RESAMPLES,
        "seed": RESAMPLE_SEED,
        "ci_level": CI_LEVEL,
        "n_slides": n_slides,
        "rule": "stratified by slide label (carcinoma / no carcinoma), percentile CI of the seed mean",
    }
    return out


def draw_row(ax: plt.Axes, y: float, row: pd.Series, color: str) -> None:
    """The CI as a thick bar without caps, the point estimate on it, the seeds as ticks above."""
    ax.plot(
        [row["ci_lo"], row["ci_hi"]],
        [y, y],
        color=color,
        lw=CI_LW,
        solid_capstyle="butt",
        zorder=3,
    )
    ax.plot(
        row["point"], y, "o", ms=POINT_SIZE, mfc=color, mec="white", mew=0.7, zorder=4
    )
    seeds = [row[f"seed_{k}"] for k in (1, 2, 3)]
    ax.plot(
        seeds,
        [y - SEED_DY] * len(seeds),  # the y axis runs top to bottom
        "|",
        ms=SEED_TICK,
        mew=SEED_MEW,
        color=color,
        ls="none",
        zorder=3,
    )


def sources_figure(rows: pd.DataFrame) -> plt.Figure:
    with plt.rc_context(
        {
            **RC,
            "axes.edgecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "axes.labelcolor": INK,
        }
    ):
        fig = plt.figure(figsize=(TEXT_W_IN, FIG_H_IN))
        top = BOTTOM + HEIGHT
        for i, ((metric, xlim, xticks), (left, width)) in enumerate(
            zip(PANELS, AXES, strict=True)
        ):
            ax = fig.add_axes((left, BOTTOM, width, HEIGHT))
            values_x = left + width + VALUES_DX
            values = blended_transform_factory(fig.transFigure, ax.transData)
            for y, (name, _, color) in enumerate(MODELS):
                row = rows.loc[(metric, name)]
                draw_row(ax, y, row, color)
                ax.text(
                    values_x,
                    y - 0.03,
                    bold(f"{row['point']:.3f}"),
                    transform=values,
                    ha="center",
                    va="bottom",
                    color=INK,
                    fontsize=FONT_SIZE - 1.5,
                )
                ax.text(
                    values_x,
                    y + 0.03,
                    f"[{row['ci_lo']:.3f}, {row['ci_hi']:.3f}]",
                    transform=values,
                    ha="center",
                    va="top",
                    color=MUTED,
                    fontsize=FONT_SIZE - 1.5,
                )
            fig.text(
                values_x,
                top + 0.005,
                f"{metric}\n[95% CI]",
                ha="center",
                va="bottom",
                color=FAINT,
                fontsize=FONT_SIZE - 2,
                linespacing=1.0,
            )
            labels = [label for _, label, _ in MODELS] if i == 0 else []
            ax.set_yticks(range(len(MODELS)), labels)
            ax.set_ylim(len(MODELS) - 0.45, -0.6)
            ax.tick_params(axis="y", length=0, pad=6, labelsize=FONT_SIZE - 0.5)
            ax.tick_params(axis="x", length=3, width=0.6, labelsize=FONT_SIZE - 0.5)
            ax.set_xlim(*xlim)
            ax.set_xticks(xticks)
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            ax.set_xlabel(bold(metric), labelpad=4)
            ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_linewidth(0.6)
            letter_x = (left - LABELS_W if i == 0 else left) + LETTER_DX[i]
            fig.text(
                letter_x,
                top + 0.04,
                f"({'ab'[i]})",
                ha="left",
                va="bottom",
                color=INK,
                fontsize=FONT_SIZE,
            )
    return fig


def code_commit() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--", str(Path(__file__).relative_to(REPO))],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return commit + ("-dirty" if dirty else "")


def log_run(stats_path: Path, out_dir: Path, stats: dict) -> str:
    """A plain MlflowClient run (no autolog, so no git remote URL is copied into the tags)."""
    from mlflow.tracking import MlflowClient
    from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID, MLFLOW_RUN_NAME

    commit = code_commit()
    assert not commit.endswith("-dirty"), "commit the script before logging"
    client = MlflowClient()
    run = client.create_run(
        EXPERIMENT_ID,
        tags={
            MLFLOW_PARENT_RUN_ID: PARENT_RUN_ID,
            MLFLOW_RUN_NAME: "fig_res_sources",
            "code_commit": commit,
            "figure": "fig:res-sources",
            "dataset": "PANDA Radboud test, all nuclei",
            **SOURCE_RUNS,
        },
    )
    run_id = run.info.run_id
    for key, value in stats["bootstrap"].items():
        client.log_param(run_id, f"bootstrap_{key}", value)
    for key, value in stats.items():
        if key == "bootstrap":
            continue
        for field in ("point", "ci_lo", "ci_hi"):
            client.log_metric(run_id, f"{key}/{field}", value[field])
        for k, s in enumerate(value["seeds"], start=1):
            client.log_metric(run_id, f"{key}/seed_{k}", s)
    for ext in ("pdf", "png"):
        client.log_artifact(run_id, str(out_dir / f"{STEM}.{ext}"), "figures")
    client.log_artifact(run_id, str(stats_path))
    client.log_artifact(run_id, str(Path(__file__)), "code")
    client.set_terminated(run_id)
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=RESULTS_DIR, type=Path)
    parser.add_argument("--check-dir", default=CHECK_DIR, type=Path)
    parser.add_argument("--resamples", default=RESAMPLES, type=Path)
    parser.add_argument("--out-dir", default=OUT_DIR, type=Path)
    parser.add_argument("--log", action="store_true")
    args = parser.parse_args()

    stats = check(args.results_dir, args.check_dir, args.resamples)
    save_figure(sources_figure(load_rows(args.results_dir)), args.out_dir, STEM)
    stats_path = args.out_dir / f"{STEM}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=1))
    print(f"{args.out_dir}/{STEM}.pdf\n{args.out_dir}/{STEM}.png\n{stats_path}")
    if args.log:
        print("MLflow run", log_run(stats_path, args.out_dir, stats))


if __name__ == "__main__":
    main()
