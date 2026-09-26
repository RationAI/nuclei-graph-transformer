"""Flags outlier slides in a QC run's `qc_metrics.csv` and plots them.

For each QC check (blur, residual artifacts, folding), an outlier threshold is
computed as Q3 + `--iqr-k` * IQR of `mean_coverage`. Slides above the threshold
for a given check are flagged for it; slides flagged on two or more checks are
reported separately as compounding-defect candidates.

Usage:
    uv run -m preprocessing.qc_outlier_analysis --run-id <run_id>
"""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient

CHECKS = [
    ("Piqe", "Blur"),
    ("ResidualArtifactsAndCoverage", "Residual Artifacts"),
    ("FoldingFunction", "Folding"),
]

INK = "#0b0b0b"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SEQUENTIAL = "#256abf"
STATUS = {1: "#fab219", 2: "#ec835a", 3: "#d03b3b"}  # warning / serious / critical

matplotlib.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)


def load_metrics(run_id: str, experiment_id: str) -> pd.DataFrame:
    uri = f"mlflow-artifacts:/{experiment_id}/{run_id}/artifacts/qc_metrics.csv"
    return pd.read_csv(download_artifacts(uri))


def outlier_threshold(values: pd.Series, iqr_k: float) -> float:
    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    if iqr == 0:
        return values.quantile(0.95)
    return q3 + iqr_k * iqr


def style_axes(ax) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS)
    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def plot_check_outliers(
    df: pd.DataFrame, column: str, label: str, top_n: int, iqr_k: float, output_dir: Path
) -> float:
    col = f"mean_coverage({column})"
    values = df[col]
    threshold = outlier_threshold(values, iqr_k)

    fig, (ax_hist, ax_rank) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax_hist.hist(values, bins=40, color=SEQUENTIAL, edgecolor=SURFACE, linewidth=0.5, zorder=2)
    ax_hist.axvline(threshold, color=STATUS[3], linestyle="--", linewidth=1.5, zorder=3)
    ax_hist.text(
        threshold,
        ax_hist.get_ylim()[1] * 0.97,
        f" outlier threshold: {threshold:.3f}",
        color=STATUS[3],
        va="top",
        fontsize=9,
    )
    ax_hist.set_xlabel("mean coverage")
    ax_hist.set_ylabel("slide count")
    ax_hist.set_title(f"{label}: distribution across all slides", loc="left", color=INK, fontsize=11)
    style_axes(ax_hist)

    top = df.nlargest(top_n, col).iloc[::-1]
    colors = [STATUS[3] if v > threshold else SEQUENTIAL for v in top[col]]
    ax_rank.barh(top["slide_name"], top[col], color=colors, zorder=2)
    for y, v in enumerate(top[col]):
        ax_rank.text(v, y, f" {v:.3f}", va="center", fontsize=8, color=INK)
    ax_rank.set_xlabel("mean coverage")
    ax_rank.set_title(f"{label}: top {top_n} slides", loc="left", color=INK, fontsize=11)
    ax_rank.tick_params(axis="y", labelsize=8)
    style_axes(ax_rank)

    fig.tight_layout()
    fig.savefig(output_dir / f"outliers_{column.lower()}.png")
    plt.close(fig)
    return threshold


def plot_multi_check_summary(
    df: pd.DataFrame, thresholds: dict, top_n: int, output_dir: Path
) -> None:
    flagged = pd.DataFrame({"slide_name": df["slide_name"]})
    for column, _ in CHECKS:
        col = f"mean_coverage({column})"
        flagged[column] = df[col] > thresholds[column]
    flagged["flagged_count"] = flagged[[c for c, _ in CHECKS]].sum(axis=1)

    multi_all = flagged[flagged["flagged_count"] >= 2].sort_values(
        "flagged_count", ascending=False
    )
    if multi_all.empty:
        return

    multi_all.to_csv(output_dir / "multi_check_outliers.csv", index=False)

    multi = multi_all.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.35 * len(multi))))
    colors = [STATUS[min(c, 3)] for c in multi["flagged_count"]]
    ax.barh(multi["slide_name"], multi["flagged_count"], color=colors, zorder=2)
    ax.set_xlabel("QC checks flagged (of 3)")
    ax.set_xlim(0, 3.5)
    ax.set_xticks([1, 2, 3])
    title = f"Slides flagged on 2+ QC checks (top {len(multi)} of {len(multi_all)})"
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.tick_params(axis="y", labelsize=8)
    style_axes(ax)

    fig.tight_layout()
    fig.savefig(output_dir / "outliers_multi_check.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="MLflow run id containing qc_metrics.csv")
    parser.add_argument(
        "--experiment-id", default="37", help="MLflow experiment id holding the run (default: 37, public cloud)"
    )
    parser.add_argument("--top-n", type=int, default=15, help="Slides shown per check's ranking plot")
    parser.add_argument(
        "--iqr-k", type=float, default=1.5, help="IQR multiplier defining the outlier threshold (Q3 + k*IQR)"
    )
    parser.add_argument(
        "--artifact-path", default="qc_outlier_analysis", help="Subdirectory inside the run's artifacts"
    )
    args = parser.parse_args()

    df = load_metrics(args.run_id, args.experiment_id)

    with TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir)
        thresholds = {}
        for column, label in CHECKS:
            col = f"mean_coverage({column})"
            if col not in df.columns:
                continue
            thresholds[column] = plot_check_outliers(
                df, column, label, args.top_n, args.iqr_k, output_dir
            )

        plot_multi_check_summary(df, thresholds, args.top_n, output_dir)

        MlflowClient().log_artifacts(args.run_id, str(output_dir), artifact_path=args.artifact_path)

    print(f"Logged outlier plots to run {args.run_id} under artifacts/{args.artifact_path}/")


if __name__ == "__main__":
    main()
