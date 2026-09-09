"""Generates a Faceted Dumbbell (Connected Dot) Plot from MLflow CSV artifacts.

Built for Q2: "How much predictive signal exists purely in a spatial-agnostic
'bag of cells' compared to their structured arrangement?"

Groups metrics into a grid (Rows = Metrics, Cols = Datasets). The Y-axis
represents modalities; a horizontal line connects the Bag-of-Cells and
Structured points for each modality, highlighting the performance gap.
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import matplotlib
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

# Project-standard palette
NAVY = "#28374A"
SHAPE_BLUE = "#3D6B8C"
GRID = "#E5E1D8"
BAG_COLOR = "#C25936"  
STRUCT_COLORS = {"AUROC": NAVY, "AUPRC": SHAPE_BLUE}

_VALID_CONDITIONS = {"bag of cells", "structured"}


def load_plot_data(inputs_config) -> pd.DataFrame:
    records = []
    for item in inputs_config:
        label = item.get("label", "Unknown")
        condition = str(item.get("condition", "")).strip()
        dataset = item.get("dataset", "MMCI")
        uri = item.get("uri")
        query = item.get("query")

        if condition.lower() not in _VALID_CONDITIONS:
            raise ValueError(
                f"Input {label!r} (uri={uri!r}) has condition={condition!r}, "
                f"expected one of {_VALID_CONDITIONS!r} explicitly."
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
            record["Condition"] = "Bag of Cells" if condition.lower() == "bag of cells" else "Structured"
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


def create_faceted_dumbbell_plot(
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
    y_pos = np.arange(len(modalities))

    n_rows = len(present_metrics)
    n_cols = len(datasets)
    
    # Reduced figure height (2.5 per row) to compress the space between modalities
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.5 * n_cols, 2.5 * n_rows), sharey=True)

    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    for row_idx, metric in enumerate(present_metrics):
        struct_color = STRUCT_COLORS.get(metric, SHAPE_BLUE)

        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            subset = df[df["Dataset"] == dataset]

            bag_data = subset[subset["Condition"] == "Bag of Cells"].set_index("Modality")
            struct_data = subset[subset["Condition"] == "Structured"].set_index("Modality")

            for j, mod in enumerate(modalities):
                bag_row = bag_data.loc[[mod]].iloc[0] if mod in bag_data.index else None
                struct_row = struct_data.loc[[mod]].iloc[0] if mod in struct_data.index else None

                val_bag = val_struct = np.nan
                if bag_row is not None:
                    val_bag, lo_bag, hi_bag = _get_point_and_ci(bag_row, metric)
                if struct_row is not None:
                    val_struct, lo_struct, hi_struct = _get_point_and_ci(struct_row, metric)

                # Connecting line — pushed to back, slightly thinner
                if not np.isnan(val_bag) and not np.isnan(val_struct):
                    ax.hlines(
                        y=y_pos[j], xmin=min(val_bag, val_struct), xmax=max(val_bag, val_struct),
                        color="#B0B0B0", linewidth=2.5, zorder=1,
                    )

                # Bag of Cells: Error bar with caps, point, and floating text
                if not np.isnan(val_bag):
                    err_lo = max(0, val_bag - lo_bag)
                    err_hi = max(0, hi_bag - val_bag)
                    ax.errorbar(val_bag, y_pos[j], xerr=[[err_lo], [err_hi]], 
                                fmt='none', ecolor=BAG_COLOR, capsize=6, capthick=2, linewidth=2, zorder=1.5)
                    
                    ax.scatter(val_bag, y_pos[j], color=BAG_COLOR, s=120, zorder=2,
                               edgecolor="white", linewidth=1.0)
                    
                    ax.annotate(f"{val_bag:.3f}", xy=(val_bag, y_pos[j]), xytext=(0, 12),
                                textcoords="offset points", ha="center", va="bottom", 
                                fontsize=9, fontfamily="monospace", color=BAG_COLOR, fontweight="bold")

                # Structured: Error bar with caps, point, and floating text
                if not np.isnan(val_struct):
                    err_lo = max(0, val_struct - lo_struct)
                    err_hi = max(0, hi_struct - val_struct)
                    ax.errorbar(val_struct, y_pos[j], xerr=[[err_lo], [err_hi]], 
                                fmt='none', ecolor=struct_color, capsize=6, capthick=2, linewidth=2, zorder=1.5)
                    
                    ax.scatter(val_struct, y_pos[j], color=struct_color, s=120, zorder=2,
                               edgecolor="white", linewidth=1.0)
                    
                    ax.annotate(f"{val_struct:.3f}", xy=(val_struct, y_pos[j]), xytext=(0, 12),
                                textcoords="offset points", ha="center", va="bottom", 
                                fontsize=9, fontfamily="monospace", color=struct_color, fontweight="bold")

            ax.set_yticks(y_pos)
            ax.set_yticklabels(modalities, fontsize=12)
            
            # Tighter vertical margins
            ax.margins(y=0.15)

            ax.set_xlabel(f"{metric} (95% CI)", fontsize=13, fontweight="bold")
            ax.grid(axis="x", color=GRID, linewidth=1.0, linestyle="--", alpha=0.9)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)

            if row_idx == 0:
                ax.set_title(f"Dataset: {dataset}", fontsize=15, color=NAVY, fontweight="bold", pad=15)

        # Lock every dataset panel for this metric onto the same x-range —
        # each panel autoscaled independently above, so without this the
        # same gap between Bag of Cells and Structured could look bigger or
        # smaller purely because of a differently-scaled panel next to it.
        row_axes = axes[row_idx, :]
        shared_xlim = (
            min(ax.get_xlim()[0] for ax in row_axes),
            max(ax.get_xlim()[1] for ax in row_axes),
        )
        for ax in row_axes:
            ax.set_xlim(shared_xlim)

    axes[0, 0].invert_yaxis()

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BAG_COLOR,
               markeredgecolor="white", markersize=12, label="Bag of Cells"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=NAVY,
               markeredgecolor="white", markersize=12, label="Structured"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=2, fontsize=12)
    fig.suptitle("Bag of Cells and Structured Arrangement Comparison", fontsize=17, color=NAVY,
                 fontweight="bold", y=1.15)
    
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved faceted dumbbell plot to {output_path}")


@with_cli_args(["+postprocessing=plots/dumbbell"])
@hydra.main(config_path="../../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    if config.get("mlflow_tracking_uri"):
        import mlflow
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    inputs = OmegaConf.to_object(config.get("inputs", []))
    metrics = OmegaConf.to_object(config.get("metrics", ["AUROC", "AUPRC"]))
    modality_order = config.get("modality_order", ["Shape", "Texture", "Spatial Stats"])
    if OmegaConf.is_config(modality_order):
        modality_order = OmegaConf.to_object(modality_order)

    if not inputs:
        raise ValueError("No inputs provided. Please define 'inputs' in your Hydra config.")

    df = load_plot_data(inputs)
    if df.empty:
        print("No data extracted. Exiting.", file=sys.stderr)
        return

    with TemporaryDirectory() as output_dir:
        out_path = Path(output_dir) / "dumbbell_q2.png"
        create_faceted_dumbbell_plot(df, metrics, modality_order, out_path)
        logger.log_artifacts(
            local_dir=str(output_dir),
            artifact_path=config.get("mlflow_artifact_path", "plots"),
        )


if __name__ == "__main__":
    main()