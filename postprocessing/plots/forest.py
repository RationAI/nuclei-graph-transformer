"""Generates a Faceted Forest Plot from MLflow CSV artifacts.

Fetches CSV files and groups them by Dataset (columns) and Metric (rows).
Exact 3-decimal metric values are aligned in a text column next to each plot.
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
        
        print(f"Fetching: {dataset} | {label} ({condition})")
        try:
            local_path = download_artifacts(uri)
            df = pd.read_csv(local_path)
            if query:
                df = df.query(query)
            if df.empty:
                print(f"  -> WARNING: No rows matched '{query}' in {uri}", file=sys.stderr)
                continue
                
            record = df.iloc[0].to_dict()
            record["Modality"] = label
            record["Condition"] = condition
            record["Dataset"] = dataset
            
            # Format the display name, omitting the condition if it is 'None'
            if str(condition).lower() in ["none", "", "default"]:
                record["Display_Name"] = label
            else:
                record["Display_Name"] = f"{label}\n({condition})"
                
            records.append(record)
            
        except Exception as e:
            print(f"  -> ERROR fetching {uri}: {e}", file=sys.stderr)
            
    return pd.DataFrame(records)

def create_faceted_forest_plot(df: pd.DataFrame, metrics: list[str], output_path: Path):
    present_metrics = [m for m in metrics if m in df.columns]
    datasets = df["Dataset"].unique()
    
    if not present_metrics or df.empty:
        print("No valid metrics found to plot.", file=sys.stderr)
        return

    modality_order = [
        "Blank Token", 
        "Spatial Stats", 
        "Shape", 
        "Shape + Spatial Stats", 
        "Shape + Relative Positions",
        "Texture"
    ]
    condition_order = ["Bag of Cells", "Structured", "Default"]
    
    df["Modality"] = pd.Categorical(df["Modality"], categories=modality_order, ordered=True)
    df["Condition"] = pd.Categorical(df["Condition"], categories=condition_order, ordered=True)
    
    df = df.dropna(subset=["Modality"])
    df = df.sort_values(by=["Modality", "Condition"], ascending=[True, True])
    
    unique_models = df["Display_Name"].unique()
    y_pos = np.arange(len(unique_models))
    
    n_rows = len(present_metrics)
    n_cols = len(datasets)
    fig_height = max(4.0, len(unique_models) * 0.8) * n_rows
    
    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=(8.0 * n_cols, fig_height), 
        sharey=True
    )
    
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes[np.newaxis, :]
    elif n_cols == 1:
        axes = axes[:, np.newaxis]

    fallback_colors = ['#1f77b4', '#2ca02c', '#d62728', '#9467bd']

    for row_idx, metric in enumerate(present_metrics):
        
        if metric.upper() == "AUROC":
            cond_colors = {'Bag of Cells': '#7f7f7f', 'Structured': '#1f77b4'}  # Blue for AUROC
        elif metric.upper() == "AUPRC":
            cond_colors = {'Bag of Cells': '#7f7f7f', 'Structured': '#ff7f0e'}  # Orange for AUPRC
        else:
            cond_colors = {'Bag of Cells': '#7f7f7f', 'Structured': '#2ca02c'}  # Green fallback

        for col_idx, dataset in enumerate(datasets):
            ax = axes[row_idx, col_idx]
            subset = df[df["Dataset"] == dataset].set_index("Display_Name")
            
            vals, lo, hi, point_colors = [], [], [], []
            for model in unique_models:
                if model in subset.index:
                    row_data = subset.loc[model]
                    if isinstance(row_data, pd.DataFrame):
                        row_data = row_data.iloc[0]
                    vals.append(row_data[metric])
                    lo.append(row_data.get(f"{metric}_lo", row_data[metric]))
                    hi.append(row_data.get(f"{metric}_hi", row_data[metric]))
                    point_colors.append(cond_colors.get(row_data["Condition"], fallback_colors[row_idx % len(fallback_colors)]))
                else:
                    vals.append(np.nan)
                    lo.append(np.nan)
                    hi.append(np.nan)
                    point_colors.append('#000000')

            vals = np.array(vals, dtype=float)
            lo = np.array(lo, dtype=float)
            hi = np.array(hi, dtype=float)
            
            xerr_lo = np.where(np.isnan(lo), 0, vals - lo)
            xerr_hi = np.where(np.isnan(hi), 0, hi - vals)
            
            for j in range(len(vals)):
                if not np.isnan(vals[j]):
                    ax.errorbar(
                        vals[j], y_pos[j], xerr=[[xerr_lo[j]], [xerr_hi[j]]], fmt='o',
                        color=point_colors[j], ecolor=point_colors[j],
                        capsize=5, markersize=8, linewidth=2
                    )
                    
                    label_text = f"{vals[j]:.3f}\n[{lo[j]:.3f}, {hi[j]:.3f}]"
                    ax.text(
                        1.15, y_pos[j],
                        label_text, 
                        transform=ax.get_yaxis_transform(), 
                        ha='center', va='center', 
                        fontsize=10, color='#333333', fontfamily='monospace'
                    )

            if row_idx == 0:
                ax.set_title(f"Dataset: {dataset}", fontsize=14, fontweight='bold', pad=15)
                ax.text(
                    1.15, 1.02, 
                    "Value\n[95% CI]", 
                    transform=ax.transAxes,
                    ha='center', va='bottom', 
                    fontsize=10, fontweight='bold', color='#333333'
                )

            ax.set_xlabel(f"{metric} (95% CI)", fontsize=12, fontweight='bold')
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
            
            min_x = np.nanmin(lo) if not np.isnan(lo).all() else 0
            max_x = np.nanmax(hi) if not np.isnan(hi).all() else 1
            pad = (max_x - min_x) * 0.15 if max_x > min_x else 0.05
            ax.set_xlim(min_x - pad, max_x + pad)
            
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4))
            ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))

    axes[0, 0].set_yticks(y_pos)
    axes[0, 0].set_yticklabels(unique_models, fontsize=11)
    
    axes[0, 0].invert_yaxis()
    
    fig.suptitle("Performance by Modality", fontsize=16, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.45, hspace=0.3, right=0.92) 
    
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved Faceted Forest Plot to {output_path}")

@with_cli_args(["+postprocessing=plots/forest"])
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
        out_path = Path(output_dir) / "forest_faceted_q2.png"
        create_faceted_forest_plot(df, metrics, out_path)
        logger.log_artifacts(local_dir=str(output_dir), artifact_path=config.get("mlflow_artifact_path", "plots"))

if __name__ == "__main__":
    main()