"""Generates a Faceted Grouped Bar Chart with CI error bars from MLflow.

Groups metrics into a grid (Rows = Metrics, Cols = Datasets).
The X-axis represents modalities (e.g., Shape, Texture), with paired bars 
showing the conditions (e.g., Bag of Cells vs Structured).
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import matplotlib
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
    'savefig.bbox': 'tight'
})

def load_plot_data(inputs_config) -> pd.DataFrame:
    records = []
    for item in inputs_config:
        label = item.get("label", "Unknown")
        condition = item.get("condition", "Default")
        dataset = item.get("dataset", "MMCI")
        uri = item.get("uri")
        query = item.get("query")
        
        try:
            local_path = download_artifacts(uri)
            df = pd.read_csv(local_path)
            if query:
                df = df.query(query)
            if df.empty:
                continue
            record = df.iloc[0].to_dict()
            record["Modality"] = label
            record["Condition"] = condition
            record["Dataset"] = dataset
            records.append(record)
        except Exception as e:
            print(f"  -> ERROR fetching {uri}: {e}", file=sys.stderr)
    return pd.DataFrame(records)

def create_faceted_bar_chart(df: pd.DataFrame, metrics: list[str], output_path: Path):
    present_metrics = [m for m in metrics if m in df.columns]
    datasets = df["Dataset"].unique()
    modalities = df["Modality"].unique()
    conditions = df["Condition"].unique()
    
    if not present_metrics or df.empty:
        print("No valid metrics found.", file=sys.stderr)
        return

    n_rows = len(present_metrics)
    n_cols = len(datasets)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.0 * n_cols, 4.5 * n_rows), sharex=True)
    
    if n_rows == 1 and n_cols == 1: axes = np.array([[axes]])
    elif n_rows == 1: axes = axes[np.newaxis, :]
    elif n_cols == 1: axes = axes[:, np.newaxis]

    x_indices = np.arange(len(modalities))
    bar_width = 0.35
    colors = {'Bag of Cells': '#a0a0a0', 'Structured': '#3D6B8C'}
    fallback_colors = ['#1f77b4', '#ff7f0e']

    for row_idx, metric in enumerate(present_metrics):
        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            subset = df[df["Dataset"] == dataset]
            
            for cond_idx, condition in enumerate(conditions):
                cond_data = subset[subset["Condition"] == condition].set_index("Modality")
                
                vals, lo, hi = [], [], []
                for mod in modalities:
                    if mod in cond_data.index:
                        row_data = cond_data.loc[mod]
                        if isinstance(row_data, pd.DataFrame): row_data = row_data.iloc[0]
                        vals.append(row_data[metric])
                        lo.append(row_data.get(f"{metric}_lo", row_data[metric]))
                        hi.append(row_data.get(f"{metric}_hi", row_data[metric]))
                    else:
                        vals.append(np.nan)
                        lo.append(np.nan)
                        hi.append(np.nan)
                
                vals = np.array(vals, dtype=float)
                lo = np.array(lo, dtype=float)
                hi = np.array(hi, dtype=float)
                
                xerr_lo = np.where(np.isnan(lo), 0, vals - lo)
                xerr_hi = np.where(np.isnan(hi), 0, hi - vals)
                
                offset = (cond_idx - len(conditions) / 2 + 0.5) * bar_width
                color = colors.get(condition, fallback_colors[cond_idx % len(fallback_colors)])
                
                rects = ax.bar(
                    x_indices + offset, vals, bar_width, 
                    label=condition if row_idx == 0 and col_idx == 0 else "", 
                    color=color, edgecolor='white', linewidth=1,
                    yerr=[xerr_lo, xerr_hi], capsize=4, error_kw={'elinewidth': 1.5, 'alpha': 0.7}
                )
                
                # Annotate top of bars
                for j, rect in enumerate(rects):
                    height = vals[j]
                    if not np.isnan(height):
                        ax.annotate(
                            f'{height:.3f}',
                            xy=(rect.get_x() + rect.get_width() / 2, hi[j] if not np.isnan(hi[j]) else height),
                            xytext=(0, 5), textcoords="offset points",
                            ha='center', va='bottom', rotation=90, fontsize=9, fontfamily='monospace'
                        )

            ax.set_ylabel(metric, fontsize=12, fontweight='bold')
            ax.set_ylim(0.5, 1.15) # Scaled so bars don't start at 0, revealing the gap size
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            
            if row_idx == 0:
                ax.set_title(f"Dataset: {dataset}", fontsize=14, fontweight='bold', pad=15)
            if row_idx == n_rows - 1:
                ax.set_xticks(x_indices)
                ax.set_xticklabels(modalities, fontsize=12, fontweight='bold')

    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(conditions), fontsize=12)
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved Faceted Bar Chart to {output_path}")

@with_cli_args(["+postprocessing=plots/bars"])
@hydra.main(config_path="../../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    if config.get("mlflow_tracking_uri"):
        import mlflow
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    inputs = OmegaConf.to_object(config.get("inputs", []))
    metrics = OmegaConf.to_object(config.get("metrics", ["AUROC", "AUPRC"]))
    if not inputs:
        raise ValueError("No inputs provided.")
    df = load_plot_data(inputs)
    if df.empty:
        return
    with TemporaryDirectory() as output_dir:
        out_path = Path(output_dir) / "bars_faceted_q2.png"
        create_faceted_bar_chart(df, metrics, out_path)
        logger.log_artifacts(local_dir=str(output_dir), artifact_path=config.get("mlflow_artifact_path", "plots"))

if __name__ == "__main__":
    main()