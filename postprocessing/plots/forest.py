"""Builds forest plots (point + 95% CI whisker, one row per configuration)
from result CSVs previously logged to MLflow by the results-loading script.

Each input URI should point to one CSV produced by that script — i.e. a
table with a "config" column, individual label columns (PE, Attention
Pattern, k, ...), fetched metric columns (AUROC, AUPRC, ...) and their
"_lo"/"_hi" CI-bound counterparts where available.

Before plotting, each table is filtered down to the standard architecture
(RoPE + k-NN) — the fixed default every other axis comparison in this
project holds constant. This is skipped automatically for tables that don't
carry PE/Attention columns at all (e.g. a k-sweep table, where the
architecture is already fixed to RoPE + k-NN for every row by construction,
not represented as a column).
"""

from tempfile import TemporaryDirectory
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

NAVY = "#28374A"
SHAPE_BLUE = "#3D6B8C"
ARRANGEMENT_GREEN = "#748A5E"
GRID = "#E5E1D8"
METRIC_COLORS = {"AUROC": NAVY, "AUPRC": SHAPE_BLUE}

# --------------------------------------------------------------------------
# Standard-architecture (RoPE + k-NN) filtering
# --------------------------------------------------------------------------

# Values seen across this project's tables meaning "RoPE is present, in its
# standard (Q/K-only) form" — deliberately excludes V-Rotated and
# Relative-Position Value Attention, which are NOT the standard architecture
# even though both also involve RoPE.
_ROPE_POSITIVE = ("rope",)
_ROPE_EXCLUDE = ("v-rotat", "relative", "none")

# Values meaning "k-NN / local", excluding dense.
_KNN_POSITIVE = ("k-nn", "knn", "local")
_KNN_EXCLUDE = ("dense",)


def _find_column(cols: list[str], must_contain: tuple[str, ...]) -> str | None:
    for c in cols:
        cl = c.lower()
        if any(term in cl for term in must_contain):
            return c
    return None


def _matches(value: str, positive: tuple[str, ...], exclude: tuple[str, ...]) -> bool:
    v = str(value).lower()
    if any(term in v for term in exclude):
        return False
    return any(term in v for term in positive)


def select_standard_architecture(df: pd.DataFrame, table_name: str = "") -> pd.DataFrame:
    """Filters a table down to the standard RoPE + k-NN architecture rows.

    Tables with no PE/Attention-like column at all (e.g. a k-sweep table,
    where every row already IS the standard architecture by construction)
    are returned unchanged — there is nothing to filter on.
    """
    cols = list(df.columns)
    pe_col = _find_column(cols, ("pe", "positional"))
    attn_col = _find_column(cols, ("attention",))

    if pe_col is None and attn_col is None:
        print(f"[{table_name}] no PE/Attention columns found — "
              f"assuming standard architecture applies to all {len(df)} row(s)")
        return df

    mask = pd.Series(True, index=df.index)
    if pe_col is not None:
        mask &= df[pe_col].map(lambda v: _matches(v, _ROPE_POSITIVE, _ROPE_EXCLUDE))
    if attn_col is not None:
        mask &= df[attn_col].map(lambda v: _matches(v, _KNN_POSITIVE, _KNN_EXCLUDE))

    filtered = df[mask]
    print(f"[{table_name}] standard-architecture filter (pe_col={pe_col!r}, "
          f"attn_col={attn_col!r}): {len(df)} -> {len(filtered)} row(s)")

    if filtered.empty:
        print(f"[{table_name}] WARNING: filter matched zero rows — "
              f"check PE/Attention value spellings in this table")

    return filtered


# --------------------------------------------------------------------------
# Plotting (unchanged)
# --------------------------------------------------------------------------

def plot_forest(df: pd.DataFrame, metrics: list[str], title: str, outpath: Path) -> None:
    """Forest-plot style: one row per config, point + 95% CI whisker."""
    present = [m for m in metrics if m in df.columns and df[m].notna().any()]
    if not present or df.empty:
        return

    n = len(df)
    y = np.arange(n)

    fig, axes = plt.subplots(
        1, len(present), figsize=(3.6 * len(present) + 2, max(3, n * 0.5)), sharey=True
    )
    if len(present) == 1:
        axes = [axes]

    for ax, m in zip(axes, present):
        vals = df[m].to_numpy(dtype=float)
        lo = df.get(f"{m}_lo", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
        hi = df.get(f"{m}_hi", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
        xerr_lo = np.where(np.isnan(lo), 0, vals - lo)
        xerr_hi = np.where(np.isnan(hi), 0, hi - vals)

        color = METRIC_COLORS.get(m, ARRANGEMENT_GREEN)
        ax.errorbar(
            vals, y, xerr=[xerr_lo, xerr_hi], fmt="o",
            color=color, ecolor=color, elinewidth=1.4, capsize=3, markersize=6,
        )
        ax.set_xlabel(m)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["config"], fontsize=8.5)
    axes[0].invert_yaxis()
    fig.suptitle(title, fontsize=11, color=NAVY, weight="bold")
    fig.tight_layout()
    fig.savefig(outpath, dpi=160)
    plt.close(fig)


def title_from_stem(stem: str) -> str:
    """'02_shape_based_models' -> 'Shape Based Models'."""
    parts = stem.split("_")
    if parts and parts[0].isdigit():
        parts = parts[1:]
    return " ".join(p.capitalize() for p in parts) or stem


@with_cli_args(["+postprocessing=plots/forest"])
@hydra.main(config_path="../../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    metrics = list(config.metrics)

    with TemporaryDirectory() as output_dir:
        output_dir_path = Path(output_dir)

        for uri in config.input_uris:
            csv_path = Path(download_artifacts(uri))
            df = pd.read_csv(csv_path)

            if "config" not in df.columns:
                print(f"[{csv_path.name}] no 'config' column — skipping")
                continue

            df = select_standard_architecture(df, table_name=csv_path.stem)
            if df.empty:
                continue

            title = title_from_stem(csv_path.stem)
            outpath = output_dir_path / f"forest_{csv_path.stem}.png"
            plot_forest(df, metrics=metrics, title=title, outpath=outpath)

        logger.log_artifacts(
            local_dir=str(output_dir_path),
            artifact_path=config.get("mlflow_artifact_path", "forest_plots"),
        )


if __name__ == "__main__":
    main()