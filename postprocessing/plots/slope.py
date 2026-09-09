"""Generates a Faceted Slope Chart from MLflow CSV artifacts.

Built for Q3: "Which matters more — restricting attention to a local
neighborhood, or providing explicit relative position via RoPE — and does
that answer hold up whether or not RoPE is present at all?"

Each modality is plotted as a single FOUR-point path, grouped by Attention
Pattern (both k-NN configurations, then both Dense configurations):

    RoPE+k-NN --[RoPE effect]--> None+k-NN --[BOTH change]--> RoPE+Dense --[RoPE effect]--> None+Dense
      (Attention=k-NN, only PE changes)  (Attention AND PE change at once)  (Attention=Dense, only PE changes)

Segments 1 and 3 are clean, single-variable comparisons (only PE changes,
Attention held fixed) — each is the RoPE effect size at one attention
pattern. Segment 2 is deliberately drawn dashed and faded: going from
None+k-NN to RoPE+Dense changes BOTH Attention Pattern and PE at once, so
its length isn't attributable to either mechanism — it's included only to
keep the path connected across all four points, not as a reading. Note that
under this grouping neither locality effect (k-NN vs. Dense at a fixed PE)
is a directly adjacent segment anymore — for that comparison, see
interaction_plot.py, which draws both PE settings' locality lines side by
side instead of one continuous path.

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
MODALITY_PALETTE = [NAVY, SHAPE_BLUE, "#C25936", "#5B8C5A", "#8C5B8C", "#8C7B3D"]

# Four conditions, grouped by Attention Pattern:
# RoPE+k-NN -- None+k-NN -- RoPE+Dense -- None+Dense.
# The middle segment (None+k-NN -> RoPE+Dense) changes both PE and
# Attention Pattern at once — see the module docstring — and is drawn
# differently below to flag that it isn't a single-variable comparison.
_CONDITION_KEYS = ["rope_knn", "none_knn", "rope_dense", "none_dense"]
_VALID_CONDITIONS = set(_CONDITION_KEYS)
_X_LABELS = ["RoPE + k-NN", "None + k-NN", "RoPE + Dense", "None + Dense"]
_CONDITION_LABEL = dict(zip(_CONDITION_KEYS, _X_LABELS))
_CONFOUNDED_SEGMENT = (1, 2)  # index pair where both variables change


def load_plot_data(inputs_config) -> pd.DataFrame:
    """Downloads one row per input from MLflow, tagged with its Modality,
    Condition (one of rope_knn / none_knn / rope_dense / none_dense), and
    Dataset.

    `condition` must be one of the four valid keys explicitly — no silent
    default, matching this project's established convention for any chart
    that depends on correctly attributing a row to one side of a comparison.
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


def _pava_nondecreasing(values: list[float]) -> list[float]:
    """Least-squares projection of `values` onto the nearest non-decreasing
    sequence (pool-adjacent-violators). Used by `_declutter` below."""
    stack: list[list[float]] = []  # each entry: [weighted_mean, weight]
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
    displacement from the true values. This is what keeps value/name/effect
    labels from literally overlapping when two modalities' numbers land close
    together, without needing them to be re-sorted or dropped.
    """
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return values.copy()
    order = np.argsort(values)
    sorted_vals = values[order]
    # Subtracting i*min_gap turns "must be >= min_gap apart" into a plain
    # non-decreasing constraint, solvable by isotonic regression (PAVA).
    shifted = sorted_vals - np.arange(len(sorted_vals)) * min_gap
    fitted = np.array(_pava_nondecreasing(list(shifted)))
    decluttered_sorted = fitted + np.arange(len(sorted_vals)) * min_gap
    result = np.empty_like(decluttered_sorted)
    result[order] = decluttered_sorted
    return result


def _points_to_data_y(ax, points: float) -> float:
    """Converts a length in points (font sizes, offsets) to the equivalent
    span of data units along `ax`'s y-axis, so label spacing can be specified
    in physical size regardless of each facet's own y-range/scale."""
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    try:
        renderer = ax.figure.canvas.get_renderer()
        bbox = ax.get_window_extent(renderer=renderer)
        px_per_point = ax.figure.dpi / 72.0
        return points * px_per_point * y_range / bbox.height
    except Exception:  # noqa: BLE001 — fall back to a rough estimate
        return points / 72.0 / 4.2 * y_range


_LABEL_HALO = [patheffects.withStroke(linewidth=2.8, foreground="white")]


def create_faceted_slope_chart(
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

    x_pos = np.array([0.0, 1.0, 2.0, 3.0])

    n_rows = len(present_metrics)
    n_cols = len(datasets)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9.0 * n_cols, 5.0 * n_rows), sharex=True)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for row_idx, metric in enumerate(present_metrics):
        row_data_axes = []
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            subset = df[df["Dataset"] == dataset]

            data_by_cond = {
                cond: subset[subset["Condition"] == cond].set_index("Modality")
                for cond in _CONDITION_KEYS
            }

            # Pass 1: plot lines/markers/error bars for every modality, and
            # collect their values — text is placed afterwards (pass 2) once
            # we know how every modality's numbers sit relative to each
            # other, instead of positioning each label from its own line in
            # isolation.
            plotted = []
            for mod in modalities:
                rows = {}
                missing = []
                for cond in _CONDITION_KEYS:
                    d = data_by_cond[cond]
                    if mod in d.index:
                        rows[cond] = d.loc[[mod]].iloc[0]
                    else:
                        missing.append(cond)

                if missing:
                    print(f"  -> WARNING: [{dataset}/{metric}] {mod} missing condition(s) "
                          f"{missing} — skipping this modality's line entirely, not just "
                          f"the missing segment, so a partial path is never silently drawn",
                          file=sys.stderr)
                    continue

                vals, los, his = {}, {}, {}
                for cond in _CONDITION_KEYS:
                    vals[cond], los[cond], his[cond] = _get_point_and_ci(rows[cond], metric)

                y = [vals[c] for c in _CONDITION_KEYS]
                color = modality_colors[mod]

                # Draw each segment separately so the one that changes both
                # variables at once (see _CONFOUNDED_SEGMENT / module
                # docstring) is visually distinct — dashed and faded —
                # instead of looking like an equally clean comparison.
                for i in range(len(x_pos) - 1):
                    confounded = (i, i + 1) == _CONFOUNDED_SEGMENT
                    ax.plot(x_pos[i:i + 2], y[i:i + 2], color=color,
                            linewidth=2.5, linestyle="--" if confounded else "-",
                            alpha=0.5 if confounded else 1.0, zorder=2)
                ax.plot(x_pos, y, color=color, linewidth=0,
                        marker="o", markersize=9, markerfacecolor=color,
                        markeredgecolor="white", markeredgewidth=1.0, zorder=3)

                for xi, cond in zip(x_pos, _CONDITION_KEYS):
                    err_lo = max(0, vals[cond] - los[cond])
                    err_hi = max(0, his[cond] - vals[cond])
                    ax.errorbar(xi, vals[cond], yerr=[[err_lo], [err_hi]], fmt="none",
                                ecolor=color, alpha=0.5, capsize=5, capthick=1.5,
                                linewidth=1.5, zorder=1.5)

                plotted.append({
                    "mod": mod,
                    "color": color,
                    "y": y,
                })

            ax.set_xticks(x_pos)
            ax.set_xticklabels(_X_LABELS, fontsize=11, fontweight="bold")
            ax.set_xlim(-0.9, 3.5)

            ax.set_ylabel(f"{metric} (95% CI)", fontsize=12, fontweight="bold")
            ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="--", alpha=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)

            if row_idx == 0:
                ax.set_title(f"Dataset: {dataset}", fontsize=14, color=NAVY, fontweight="bold", pad=15)

            if not plotted:
                continue
            row_data_axes.append(ax)

            # Leave headroom above/below the data for labels to be nudged
            # into, then lock in the axes' pixel geometry so point-sized
            # gaps below can be converted to this facet's data units.
            ax.margins(y=0.22)
            fig.canvas.draw()
            gap_value = _points_to_data_y(ax, 13.0)
            gap_name = _points_to_data_y(ax, 15.0)

            # Pass 2a: value labels at each of the 3 x-columns, decluttered
            # vertically within that column so close-together modalities'
            # numbers never sit on top of each other. They're also nudged
            # sideways off the marker/line itself (not centered on top of
            # it) — column 0 to the right (clear of the name labels further
            # left), columns 1 and 2 to the left. A faint leader line marks
            # the true value when a label had to be nudged away from it.
            for col_i, xi in enumerate(x_pos):
                raw_ys = np.array([p["y"][col_i] for p in plotted])
                label_ys = _declutter(raw_ys, gap_value)
                for p, raw_y, label_y in zip(plotted, raw_ys, label_ys):
                    if abs(label_y - raw_y) > gap_value * 0.25:
                        ax.plot([xi, xi], [raw_y, label_y], color=p["color"], alpha=0.35,
                                linewidth=0.7, zorder=1.8, solid_capstyle="round")
                    if col_i == 0:
                        dx, ha = 11, "left"
                    else:
                        dx, ha = -11, "right"
                    txt = ax.annotate(f"{raw_y:.3f}", xy=(xi, label_y), xytext=(dx, 0),
                                       textcoords="offset points", ha=ha, va="center",
                                       fontsize=8, fontfamily="monospace", fontweight="bold",
                                       color=p["color"], zorder=4)
                    txt.set_path_effects(_LABEL_HALO)

            # Pass 2b: modality name labels further to the left of column 0,
            # likewise decluttered vertically.
            raw_name_ys = np.array([p["y"][0] for p in plotted])
            name_ys = _declutter(raw_name_ys, gap_name)
            for p, name_y in zip(plotted, name_ys):
                txt = ax.annotate(p["mod"], xy=(x_pos[0], name_y), xytext=(-30, 0),
                                   textcoords="offset points", ha="right", va="center",
                                   fontsize=9, fontweight="bold", color=p["color"], zorder=4)
                txt.set_path_effects(_LABEL_HALO)

        # Lock every dataset panel for this metric onto the same y-range —
        # each panel autoscaled independently above, so without this the
        # same effect size could look bigger or smaller purely because of a
        # differently-scaled panel next to it.
        if row_data_axes:
            shared_ylim = (
                min(ax.get_ylim()[0] for ax in row_data_axes),
                max(ax.get_ylim()[1] for ax in row_data_axes),
            )
            for ax in axes[row_idx, :]:
                ax.set_ylim(shared_ylim)

    # Reserve top space for title + caption as figure-fraction slots sized
    # from the actual figure height (not a fixed fraction) — a one-row
    # figure needs a much bigger fraction set aside for the same two lines
    # of header text than a multi-row one does.
    fig_height_in = 5.0 * n_rows
    header_in = 0.95  # title line + gap + caption line + padding
    rect_top = max(0.72, 1 - header_in / fig_height_in)
    fig.tight_layout(rect=[0, 0, 1, rect_top])
    title_y = 1 - (0.18 + 0.20) / fig_height_in
    caption_y = 1 - (0.18 + 0.40 + 0.18) / fig_height_in
    fig.suptitle(
        "Locality vs. RoPE Effect",
        fontsize=16, color=NAVY, fontweight="bold", y=title_y,
    )
    fig.text(0.5, caption_y,
              "Dashed segment (None + k-NN → RoPE + Dense) changes both "
              "Attention Pattern and PE at once — not a single-variable effect.",
              ha="center", va="center", fontsize=9.5, color="#666666", fontstyle="italic")
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved faceted slope chart to {output_path}")


@with_cli_args(["+postprocessing=plots/slope"])
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
        out_path = Path(output_dir) / "slope_q3.png"
        create_faceted_slope_chart(df, metrics, modality_order, out_path)
        logger.log_artifacts(
            local_dir=str(output_dir),
            artifact_path=config.get("mlflow_artifact_path", "plots"),
        )


if __name__ == "__main__":
    main()