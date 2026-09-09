"""Generates a Faceted k-Sweep Chart (line + CI band) from MLflow CSV artifacts.

Built for Q4: "What is the optimal receptive-field scale for k-NN attention,
and how does it compare to full (global) attention?"

Each panel sweeps k-NN neighborhood size k against a metric, drawn as a line
with a shaded 95% CI ribbon. The Dense (global-attention) condition is
overlaid as a horizontal dashed reference line with its own shaded CI band,
rather than as a point on the k-axis — Dense has no principled x-position
(it isn't k=some-number, it's k=every-node, which varies per crop), so a
reference line sidesteps that ambiguity while still making the local-vs-
global comparison immediate: does the sweep's peak sit above, below, or
converge toward the Dense line? The peak k is marked explicitly on each
panel so the optimal scale is readable by eye, not just by squinting at the
curve.

Discipline: Positional Encoding must be fixed at standard RoPE for every
point plotted here — both the sweep and the Dense reference — so that only
Attention Pattern (the receptive-field scale) varies. Mixing in a None+Dense
point instead of RoPE+Dense would reintroduce the exact PE/pattern confound
Q3 was built to isolate: a gap would no longer be attributable to scale
alone. This is enforced the same way Q3 enforces its own condition split —
by requiring each input to declare an explicit `k`, with no silent default —
plus a best-effort warning (see `load_plot_data`) if a query's text doesn't
even mention RoPE.

Groups metrics x datasets into rows (one row per metric-dataset pair) and
modalities into columns, so each column is a small-multiples panel for one
modality.
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import matplotlib
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

matplotlib.rcParams.update({
    'figure.max_open_warning': 0,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})

NAVY = "#28374A"
SHAPE_BLUE = "#3D6B8C"
GRID = "#E5E1D8"
MODALITY_COLORS = {"Shape": NAVY, "Texture": SHAPE_BLUE}
FALLBACK_COLOR = "#5B8C5A"

_LABEL_HALO = [patheffects.withStroke(linewidth=2.8, foreground="white")]


def load_plot_data(inputs_config) -> pd.DataFrame:
    """Downloads one row per input from MLflow, tagged with its Modality,
    Dataset, k (the sweep's neighborhood size), and IsDense.

    `k` must be given explicitly per input — either an integer (a sweep
    point) or the literal string "dense" (the global-attention reference) —
    matching this project's convention of never silently defaulting a field
    that determines which side of a comparison a row belongs to.
    """
    records = []
    for item in inputs_config:
        label = item.get("label", "Unknown")
        dataset = item.get("dataset", "MMCI")
        uri = item.get("uri")
        query = item.get("query")
        k_raw = item.get("k")

        if k_raw is None:
            raise ValueError(
                f"Input {label!r} (uri={uri!r}) has no 'k' field — expected an "
                f"integer neighborhood size or the literal string 'dense'."
            )
        k_str = str(k_raw).strip().lower()
        is_dense = k_str == "dense"
        if is_dense:
            k_val = np.nan
        else:
            try:
                k_val = float(k_raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Input {label!r} (uri={uri!r}) has k={k_raw!r}, expected an "
                    f"integer or the literal string 'dense'."
                )

        if query and "rope" not in str(query).lower():
            print(f"  -> WARNING: [{label}/{dataset}] query {query!r} doesn't mention "
                  f"RoPE — Q4 requires PE fixed at standard RoPE for every point, "
                  f"including the Dense reference, or scale and position encoding "
                  f"become confounded again.", file=sys.stderr)

        try:
            local_path = download_artifacts(uri)
            df = pd.read_csv(local_path)
            if query:
                df = df.query(query)
            if df.empty:
                print(f"  -> WARNING: query '{query}' matched no rows in {uri}", file=sys.stderr)
                continue
            if len(df) > 1:
                print(f"  -> WARNING: query '{query}' matched {len(df)} rows in {uri}, "
                      f"expected exactly 1 — taking the first", file=sys.stderr)

            record = df.iloc[0].to_dict()
            record["Modality"] = label
            record["Dataset"] = dataset
            record["k"] = k_val
            record["IsDense"] = is_dense
            records.append(record)
        except Exception as e:  # noqa: BLE001 — surface, don't crash the whole batch
            print(f"  -> ERROR fetching {uri}: {e}", file=sys.stderr)

    return pd.DataFrame(records)


def _get_point_and_ci(row: pd.Series, metric: str) -> tuple[float, float, float]:
    val = row[metric]
    lo = row.get(f"{metric}_lo", val)
    hi = row.get(f"{metric}_hi", val)
    lo = val if pd.isna(lo) else lo
    hi = val if pd.isna(hi) else hi
    return float(val), float(lo), float(hi)


def create_k_sweep_chart(
    df: pd.DataFrame, metrics: list[str], modality_order: list[str], output_path: Path
) -> None:
    present_metrics = [m for m in metrics if m in df.columns]
    if not present_metrics or df.empty:
        print("No valid metrics found.", file=sys.stderr)
        return

    datasets = sorted(df["Dataset"].unique())
    df["Modality"] = pd.Categorical(df["Modality"], categories=modality_order, ordered=True)
    df = df.dropna(subset=["Modality"])

    modalities = [m for m in modality_order if m in df["Modality"].values]
    if not modalities:
        print("No modalities matched modality_order.", file=sys.stderr)
        return

    # Rows = metrics, columns = datasets — a compact grid (2x2 for the usual
    # AUROC/AUPRC x MMCI/Radboud case). Modalities are overlaid within each
    # panel (distinguished by color) rather than given their own column, so
    # panels for modalities with no data don't waste grid space.
    n_rows = len(present_metrics)
    n_cols = len(datasets)

    # Shared k-range across every panel (not just each panel's own points),
    # so a modality with a sparser sweep reads as sparse against the same
    # scale instead of silently rescaling to fill its axes.
    all_k = df.loc[~df["IsDense"], "k"].dropna()
    if not all_k.empty:
        k_min, k_max = float(all_k.min()), float(all_k.max())
        k_pad = max((k_max - k_min) * 0.08, 1.0)
        shared_xlim = (k_min - k_pad, k_max + k_pad)
    else:
        shared_xlim = None

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 4.5 * n_rows), squeeze=False)

    modality_handles: dict[str, Line2D] = {}
    dense_seen = False

    for row_idx, metric in enumerate(present_metrics):
        row_data_axes = []
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx][col_idx]
            subset_ds = df[df["Dataset"] == dataset]
            plotted_any = False

            for m_idx, modality in enumerate(modalities):
                subset = subset_ds[subset_ds["Modality"] == modality]
                sweep = subset[~subset["IsDense"]].sort_values("k")
                dense = subset[subset["IsDense"]]

                if sweep.empty and dense.empty:
                    continue
                plotted_any = True
                color = MODALITY_COLORS.get(modality, FALLBACK_COLOR)

                if not sweep.empty:
                    k_vals = sweep["k"].to_numpy(dtype=float)
                    vals, los, his = [], [], []
                    for _, row in sweep.iterrows():
                        v, lo, hi = _get_point_and_ci(row, metric)
                        vals.append(v)
                        los.append(lo)
                        his.append(hi)
                    vals, los, his = np.array(vals), np.array(los), np.array(his)

                    ax.fill_between(k_vals, los, his, color=color, alpha=0.15, zorder=1, linewidth=0)
                    line, = ax.plot(k_vals, vals, color=color, linewidth=2.2, marker="o", markersize=6,
                                     markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.0,
                                     zorder=3, label=modality)
                    modality_handles.setdefault(modality, line)

                    # Peak k: marked with a vertical dashed guide, a larger
                    # marker, and an explicit k*=... label, so the optimal
                    # scale is answerable by eye per panel. Text is staggered
                    # per modality so overlaid peaks don't collide.
                    peak_i = int(np.argmax(vals))
                    peak_k, peak_val = k_vals[peak_i], vals[peak_i]
                    ax.axvline(peak_k, color=color, linestyle=":", linewidth=1.3, alpha=0.6, zorder=1.5)
                    ax.scatter([peak_k], [peak_val], s=110, color=color, zorder=4,
                               edgecolor="white", linewidth=1.3, marker="*")
                    txt = ax.annotate(f"k*={peak_k:g}", xy=(peak_k, peak_val),
                                       xytext=(0, 12 + m_idx * 14),
                                       textcoords="offset points", ha="center", va="bottom",
                                       fontsize=8.5, fontweight="bold", color=color, zorder=5)
                    txt.set_path_effects(_LABEL_HALO)

                if not dense.empty:
                    val_dense, lo_dense, hi_dense = _get_point_and_ci(dense.iloc[0], metric)
                    ax.axhspan(lo_dense, hi_dense, color=color, alpha=0.10, zorder=0.5)
                    ax.axhline(val_dense, color=color, linestyle="--", linewidth=1.8,
                               zorder=1.2)
                    dense_seen = True
                    txt = ax.annotate(f"{modality} Dense: {val_dense:.3f}", xy=(1.0, val_dense),
                                       xycoords=("axes fraction", "data"),
                                       xytext=(-6, 6 + m_idx * 12),
                                       textcoords="offset points", ha="right", va="bottom",
                                       fontsize=8.5, fontweight="bold", color=color, zorder=5)
                    txt.set_path_effects(_LABEL_HALO)

            if not plotted_any:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=10, color="#999999")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.spines[:].set_visible(False)
                if row_idx == 0:
                    ax.set_title(dataset, fontsize=14, color=NAVY, fontweight="bold", pad=15)
                continue
            row_data_axes.append(ax)

            ax.set_xlabel("k (nearest neighbors)", fontsize=10.5, fontweight="bold")
            ax.set_ylabel(f"{dataset}\n{metric} (95% CI)", fontsize=10.5, fontweight="bold")
            ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="--", alpha=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)

            if shared_xlim is not None:
                ax.set_xlim(shared_xlim)
            if not all_k.empty:
                ax.set_xticks(sorted(all_k.unique()))

            if row_idx == 0:
                ax.set_title(dataset, fontsize=14, color=NAVY, fontweight="bold", pad=15)

        # Lock every dataset panel for this metric onto the same y-range —
        # each panel autoscaled independently above, so without this a
        # modality's sweep could look identically shaped in two datasets
        # while actually sitting at very different absolute performance.
        if row_data_axes:
            shared_ylim = (
                min(ax.get_ylim()[0] for ax in row_data_axes),
                max(ax.get_ylim()[1] for ax in row_data_axes),
            )
            for ax in row_data_axes:
                ax.set_ylim(shared_ylim)

    handles = [
        Line2D([0], [0], color=color, linewidth=2.2, marker="o", markersize=6,
               markerfacecolor=color, markeredgecolor="white", label=f"{modality} (k-NN sweep)")
        for modality, line in modality_handles.items()
        for color in [line.get_color()]
    ]
    if dense_seen:
        handles.append(
            Line2D([0], [0], color="#666666", linewidth=1.8, linestyle="--",
                   label="Dense / global attention (modality color)")
        )
    handles.append(
        Line2D([0], [0], color="#666666", linewidth=1.3, linestyle=":", marker="*",
               markersize=10, markerfacecolor="#666666", label="Peak k")
    )
    # Reserve the top of the figure for title + legend + caption, laid out
    # in fixed figure-fraction slots (not y>1 offsets), so tight_layout's
    # axes region and the header content never fight over the same space.
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    fig.suptitle("k-NN vs. Global Attention: Spatial Statistics",
                 fontsize=16, color=NAVY, fontweight="bold", y=0.985)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=len(handles), fontsize=9.5, frameon=False, columnspacing=1.3, handletextpad=0.6)


    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved faceted k-sweep chart to {output_path}")


@with_cli_args(["+postprocessing=plots/k_sweep"])
@hydra.main(config_path="../../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    if config.get("mlflow_tracking_uri"):
        import mlflow
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    inputs = OmegaConf.to_object(config.get("inputs", []))
    metrics = OmegaConf.to_object(config.get("metrics", ["AUROC", "AUPRC"]))
    modality_order = config.get("modality_order", ["Shape", "Texture"])
    if OmegaConf.is_config(modality_order):
        modality_order = OmegaConf.to_object(modality_order)

    if not inputs:
        raise ValueError("No inputs provided. Please define 'inputs' in your Hydra config.")

    df = load_plot_data(inputs)
    if df.empty:
        print("No data extracted. Exiting.", file=sys.stderr)
        return

    with TemporaryDirectory() as output_dir:
        out_path = Path(output_dir) / "k_sweep_q4.png"
        create_k_sweep_chart(df, metrics, modality_order, out_path)
        logger.log_artifacts(
            local_dir=str(output_dir),
            artifact_path=config.get("mlflow_artifact_path", "plots"),
        )


if __name__ == "__main__":
    main()
