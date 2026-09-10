"""Generates a Faceted Interaction Plot from MLflow CSV artifacts.

Built for Q3 (extended): "Which matters more — restricting attention to a
local neighborhood, or providing explicit relative position via RoPE — and
does locality's effect depend on whether RoPE is present at all?"

Each modality is plotted as up to TWO lines, both spanning k-NN -> Dense:

    RoPE line:  RoPE + k-NN  ---->  RoPE + Dense   (locality effect, PE fixed at RoPE)
    None line:  None + k-NN  ---->  None + Dense   (locality effect, PE fixed at None)

Both lines share the modality's color and differ only in line style (solid
= RoPE, dashed = None) and marker shape (circle = RoPE, square = None).

A line is plotted as long as both of its conditions (k-NN and Dense) are 
available. If a modality has data for one line but is missing data for the 
other, the available line will still be drawn.
Groups metrics into a grid (Rows = Metrics, Cols = Datasets).
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

matplotlib.use("Agg")

matplotlib.rcParams.update({
    'figure.max_open_warning': 0,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})

NAVY = "#28374A"
SHAPE_BLUE = "#3D6B8C"
GRID = "#E5E1D8"
MODALITY_PALETTE = [NAVY, SHAPE_BLUE, "#C25936", "#5B8C5A", "#8C5B8C", "#8C7B3D"]

# Four conditions, forming the two lines of the interaction plot.
_CONDITION_KEYS = ["rope_knn", "rope_dense", "none_knn", "none_dense"]
_VALID_CONDITIONS = set(_CONDITION_KEYS)
_X_LABELS = ["k-NN", "Dense"]

# Which two conditions make up each line, in x-order (k-NN, then Dense).
_LINES = {
    "RoPE": ("rope_knn", "rope_dense"),
    "None": ("none_knn", "none_dense"),
}
_LINE_STYLE = {"RoPE": {"linestyle": "-", "marker": "o"}, "None": {"linestyle": "--", "marker": "s"}}


def load_plot_data(inputs_config) -> pd.DataFrame:
    """Downloads one row per input from MLflow, tagged with its Modality,
    Condition (one of rope_knn / rope_dense / none_knn / none_dense), and
    Dataset.
    """
    records = []
    for item in inputs_config:
        label = item.get("label", "Unknown")
        condition = str(item.get("condition", "")).strip().lower()
        dataset = item.get("dataset", "MMCI")
        uri = item.get("uri")
        query = item.get("query")

        if condition not in _VALID_CONDITIONS:
            raise ValueError(
                f"Input {label!r} (uri={uri!r}) has condition={condition!r}, "
                f"expected one of {sorted(_VALID_CONDITIONS)!r} explicitly."
            )

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
            record["Condition"] = condition
            record["Dataset"] = dataset
            records.append(record)
        except Exception as e:
            print(f"  -> ERROR fetching {uri}: {e}", file=sys.stderr)

    return pd.DataFrame(records)


def _get_point_and_ci(row: pd.Series, metric: str) -> tuple[float, float, float]:
    val = row[metric]
    lo = row.get(f"{metric}_lo", val)
    hi = row.get(f"{metric}_hi", val)
    lo = val if pd.isna(lo) else lo
    hi = val if pd.isna(hi) else hi
    return float(val), float(lo), float(hi)


def _pava_nondecreasing(values: list[float]) -> list[float]:
    """Least-squares projection of `values` onto the nearest non-decreasing
    sequence (pool-adjacent-violators). Used by `_declutter` below."""
    stack: list[list[float]] = []
    for v in values:
        block = [v, 1.0]
        while stack and stack[-1][0] > block[0]:
            prev = stack.pop()
            w = prev[1] + block[1]
            block = [(prev[0] * prev[1] + block[0] * block[1]) / w, w]
        stack.append(block)
    out: list[float] = []
    for mean, w in stack:
        out.extend([mean] * int(w))
    return out


def _declutter(values: np.ndarray, min_gap: float) -> np.ndarray:
    """Nudges a set of y-values apart so that, in rank order, no two are
    closer than `min_gap` — while preserving rank order and minimizing total
    displacement from the true values."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return values.copy()
    order = np.argsort(values)
    sorted_vals = values[order]
    shifted = sorted_vals - np.arange(len(sorted_vals)) * min_gap
    fitted = np.array(_pava_nondecreasing(list(shifted)))
    decluttered_sorted = fitted + np.arange(len(sorted_vals)) * min_gap
    result = np.empty_like(decluttered_sorted)
    result[order] = decluttered_sorted
    return result


def _points_to_data_y(ax, points: float) -> float:
    """Converts a length in points to the equivalent span of data units
    along `ax`'s y-axis, so label spacing is specified in physical size
    regardless of each facet's own y-range/scale."""
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    try:
        renderer = ax.figure.canvas.get_renderer()
        bbox = ax.get_window_extent(renderer=renderer)
        px_per_point = ax.figure.dpi / 72.0
        return points * px_per_point * y_range / bbox.height
    except Exception:
        return points / 72.0 / 4.2 * y_range


_LABEL_HALO = [patheffects.withStroke(linewidth=2.8, foreground="white")]


def create_interaction_plot(
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
    modality_colors = {mod: MODALITY_PALETTE[i % len(MODALITY_PALETTE)] for i, mod in enumerate(modalities)}

    x_pos = np.array([0.0, 1.0])

    n_rows = len(present_metrics)
    n_cols = len(datasets)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 5.0 * n_rows), sharex=True)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for row_idx, metric in enumerate(present_metrics):
        row_data_axes = []
        panel_plotted = {}
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            subset = df[df["Dataset"] == dataset]

            data_by_cond = {
                cond: subset[subset["Condition"] == cond].set_index("Modality")
                for cond in _CONDITION_KEYS
            }

            plotted = []
            for mod in modalities:
                rows = {}
                for cond in _CONDITION_KEYS:
                    d = data_by_cond[cond]
                    if mod in d.index:
                        rows[cond] = d.loc[[mod]].iloc[0]

                vals, los, his = {}, {}, {}
                for cond, row in rows.items():
                    vals[cond], los[cond], his[cond] = _get_point_and_ci(row, metric)

                color = modality_colors[mod]
                line_data = {}
                for line_name, (c0, c1) in _LINES.items():
                    if c0 in vals and c1 in vals:
                        y = [vals[c0], vals[c1]]
                        style = _LINE_STYLE[line_name]
                        ax.plot(x_pos, y, color=color, linewidth=2.3,
                                marker=style["marker"], markersize=8, linestyle=style["linestyle"],
                                markerfacecolor=color, markeredgecolor="white", markeredgewidth=1.0,
                                zorder=2)
                        for xi, cond in zip(x_pos, (c0, c1)):
                            err_lo = max(0, vals[cond] - los[cond])
                            err_hi = max(0, his[cond] - vals[cond])
                            ax.errorbar(xi, vals[cond], yerr=[[err_lo], [err_hi]], fmt="none",
                                        ecolor=color, alpha=0.4, capsize=4, capthick=1.3,
                                        linewidth=1.3, zorder=1.5)
                        line_data[line_name] = {"y": y}
                    else:
                        missing_points = [c for c in (c0, c1) if c not in vals]
                        print(f"  -> INFO: [{dataset}/{metric}] {mod} missing {missing_points} "
                              f"— skipping {line_name} line.", file=sys.stderr)

                if not line_data:
                    continue  # Skip modality only if neither line could be drawn

                plotted.append({"mod": mod, "color": color, "lines": line_data, "vals": vals})

            ax.set_xticks(x_pos)
            ax.set_xticklabels(_X_LABELS, fontsize=11, fontweight="bold")
            ax.set_xlim(-0.9, 1.6)

            ax.set_ylabel(f"{metric} (95% CI)", fontsize=12, fontweight="bold")
            ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="--", alpha=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)

            if row_idx == 0:
                ax.set_title(f"Dataset: {dataset}", fontsize=14, color=NAVY, fontweight="bold", pad=15)

            if not plotted:
                continue
            row_data_axes.append(ax)
            panel_plotted[dataset] = plotted

            ax.margins(y=0.25)

        # Lock every dataset panel for this metric onto the same y-range —
        # each panel autoscaled independently above, so without this a
        # modality that looks flat in one dataset could sit on a wildly
        # different scale than the same modality one panel over. This has
        # to happen *before* the label-decluttering pass below: decluttering
        # spaces labels apart in data units based on each panel's current
        # view, so if the view were widened afterward, labels that were
        # correctly separated for the narrow (pre-share) range would end up
        # visually packed closer together once the axis stretches.
        if row_data_axes:
            shared_ylim = (
                min(ax.get_ylim()[0] for ax in row_data_axes),
                max(ax.get_ylim()[1] for ax in row_data_axes),
            )
            for ax in axes[row_idx, :]:
                ax.set_ylim(shared_ylim)
        fig.canvas.draw()

        for col_idx, dataset in enumerate(datasets):
            if dataset not in panel_plotted:
                continue
            ax = axes[row_idx, col_idx]
            plotted = panel_plotted[dataset]

            gap_value = _points_to_data_y(ax, 12.0)
            gap_name = _points_to_data_y(ax, 14.0)

            # Pass 2a: Value labels at each endpoint
            for col_i, xi in enumerate(x_pos):
                entries = []
                for p in plotted:
                    for line_name in p["lines"]:
                        entries.append((p["color"], p["lines"][line_name]["y"][col_i]))
                if not entries:
                    continue
                raw_ys = np.array([e[1] for e in entries])
                label_ys = _declutter(raw_ys, gap_value)
                dx, ha = (10, "left") if col_i == 0 else (-10, "right")
                for (color, raw_y), label_y in zip(entries, label_ys):
                    if abs(label_y - raw_y) > gap_value * 0.25:
                        ax.plot([xi, xi], [raw_y, label_y], color=color, alpha=0.3,
                                linewidth=0.7, zorder=1.8, solid_capstyle="round")
                    txt = ax.annotate(f"{raw_y:.3f}", xy=(xi, label_y), xytext=(dx, 0),
                                       textcoords="offset points", ha=ha, va="center",
                                       fontsize=7.5, fontfamily="monospace", fontweight="bold",
                                       color=color, zorder=4)
                    txt.set_path_effects(_LABEL_HALO)

            # Pass 2b: Modality name labels at left midpoint
            raw_name_ys = np.array([
                sum(p["lines"][ln]["y"][0] for ln in p["lines"]) / len(p["lines"]) for p in plotted
            ])
            name_ys = _declutter(raw_name_ys, gap_name)
            for p, name_y in zip(plotted, name_ys):
                txt = ax.annotate(p["mod"], xy=(x_pos[0], name_y), xytext=(-32, 0),
                                   textcoords="offset points", ha="right", va="center",
                                   fontsize=9, fontweight="bold", color=p["color"], zorder=4)
                txt.set_path_effects(_LABEL_HALO)

    handles = [
        Line2D([0], [0], color="#555555", linewidth=2.3, linestyle="-", marker="o",
               markersize=8, markerfacecolor="#555555", markeredgecolor="white",
               label="RoPE present"),
        Line2D([0], [0], color="#555555", linewidth=2.3, linestyle="--", marker="s",
               markersize=8, markerfacecolor="#555555", markeredgecolor="white",
               label="RoPE absent"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.04),
               ncol=2, fontsize=11, frameon=False)
    fig.suptitle(
        "Locality vs. Precise Positions",
        fontsize=14, color=NAVY, fontweight="bold", y=1.09,
    )
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved interaction plot to {output_path}")


@with_cli_args(["+postprocessing=plots/interaction_plot"])
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
        out_path = Path(output_dir) / "interaction_q3.png"
        create_interaction_plot(df, metrics, modality_order, out_path)
        logger.log_artifacts(
            local_dir=str(output_dir),
            artifact_path=config.get("mlflow_artifact_path", "plots"),
        )


if __name__ == "__main__":
    main()