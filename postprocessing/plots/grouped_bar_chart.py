"""Generates side-by-side Grouped Bar Charts per Dataset from MLflow.

Each subplot represents a Dataset, with X-axis showing Modality and Condition groups,
and bars grouped side-by-side for all selected metrics with non-overlapping horizontal labels.
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
    'savefig.dpi': 1200,
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
                print(f"  -> WARNING: [{label}/{dataset}] query '{query}' matched no rows "
                      f"in {uri} — this bar will be missing from the chart", file=sys.stderr)
                continue
            if len(df) > 1:
                print(f"  -> WARNING: [{label}/{dataset}] query '{query}' matched {len(df)} rows "
                      f"in {uri}, expected exactly 1 — taking the first", file=sys.stderr)
            record = df.iloc[0].to_dict()
            record["Modality"] = label
            
            if str(condition).lower() in ["none", "", "default"]:
                record["Condition"] = "Default"
            else:
                record["Condition"] = condition
                
            record["Dataset"] = dataset
            records.append(record)
        except Exception as e:
            print(f"  -> ERROR fetching {uri}: {e}", file=sys.stderr)
    return pd.DataFrame(records)

def create_dataset_grouped_bar_chart(
    df: pd.DataFrame, 
    metrics: list[str], 
    modality_order: list[str], 
    condition_order: list[str], 
    output_path: Path
):
    present_metrics = [m for m in metrics if m in df.columns]
    datasets = df["Dataset"].unique()
    
    if not present_metrics or df.empty:
        print("No valid metrics found.", file=sys.stderr)
        return

    if not modality_order:
        modality_order = sorted(df["Modality"].unique())
    if not condition_order:
        condition_order = sorted(df["Condition"].unique())
    
    df["Modality"] = pd.Categorical(df["Modality"], categories=modality_order, ordered=True)
    df["Condition"] = pd.Categorical(df["Condition"], categories=condition_order, ordered=True)
    
    df = df.dropna(subset=["Modality", "Condition"])
    df = df.sort_values(by=["Modality", "Condition"])
    
    modalities = [m for m in modality_order if m in df["Modality"].values]
    conditions = [c for c in condition_order if c in df["Condition"].values]
    
    x_labels = []
    x_keys = []
    for mod in modalities:
        for cond in conditions:
            sub = df[(df["Modality"] == mod) & (df["Condition"] == cond)]
            if not sub.empty:
                cond_str = f"\n({cond})" if cond != "Default" else ""
                x_labels.append(f"{mod}{cond_str}")
                x_keys.append((mod, cond))

    if not x_keys:
        print("No valid groups to plot after applying modality/condition filters.", file=sys.stderr)
        return

    # Shared y-range for the whole figure (all dataset panels), sized from
    # the actual plotted values instead of a fixed guess — otherwise bars
    # sit in the bottom half of the axes whenever the real data doesn't
    # reach the hardcoded ceiling.
    lo_series, hi_series = [], []
    for metric in present_metrics:
        vals = df[metric].astype(float)
        lo = df[f"{metric}_lo"].astype(float) if f"{metric}_lo" in df.columns else vals
        hi = df[f"{metric}_hi"].astype(float) if f"{metric}_hi" in df.columns else vals
        lo_series.append(lo.fillna(vals))
        hi_series.append(hi.fillna(vals))
    data_min = min(s.min() for s in lo_series)
    data_max = max(s.max() for s in hi_series)
    data_range = data_max - data_min if data_max > data_min else 0.1
    y_floor = max(0.0, data_min - data_range * 0.08)
    # Headroom on top for the per-bar value labels (annotated above the bar,
    # staggered further for alternating bars). Sized off data_max directly
    # rather than the full min-max range — when one metric sits much lower
    # than another (e.g. AUPRC vs. AUROC in the same panel), a range-based
    # pad would inflate the ceiling far past the actual highest bar.
    y_ceiling = data_max + max(data_range * 0.08, 0.04)

    n_metrics = len(present_metrics)
    bar_width = 0.02
    group_width = n_metrics * bar_width + 0.01
    x_indices = np.arange(len(x_keys)) * group_width
    
    metric_colors = {
        "AUROC": '#1f77b4',
        "AUPRC": '#ff7f0e',
        "Accuracy": '#2ca02c',
        "Precision": '#d62728',
        "Recall": '#9467bd',
        "Specificity": '#8c564b',
    }
    
    n_datasets = len(datasets)
    panel_width = 0.75 * len(x_keys) * n_metrics + 0.6

    # Many x-axis groups make side-by-side panels too narrow to read (labels
    # collide), so stack the datasets into rows instead once there's more
    # than a handful of groups; otherwise keep them side by side.
    stacked = len(x_keys) > 3
    if stacked:
        n_rows, n_cols = n_datasets, 1
        figsize = (panel_width, 7.5 * n_datasets)
    else:
        n_rows, n_cols = 1, n_datasets
        figsize = (panel_width * n_datasets, 7.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=True, sharex=stacked, squeeze=False)

    handles, labels = [], []
    all_lo, all_hi = [], []

    for panel_idx, dataset in enumerate(datasets):
        ax = axes[panel_idx, 0] if stacked else axes[0, panel_idx]
        subset = df[df["Dataset"] == dataset]
        
        for m_idx, metric in enumerate(present_metrics):
            vals, lo, hi = [], [], []
            for mod, cond in x_keys:
                row = subset[(subset["Modality"] == mod) & (subset["Condition"] == cond)]
                if not row.empty:
                    row_data = row.iloc[0]
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

            all_lo.extend(lo[~np.isnan(lo)])
            all_hi.extend(hi[~np.isnan(hi)])
            
            offset = (m_idx - n_metrics / 2 + 0.5) * bar_width
            color = metric_colors.get(metric, '#333333')
            
            bar_label = metric if panel_idx == 0 else ""

            rects = ax.bar(
                x_indices + offset, vals, bar_width,
                label=bar_label,
                color=color, edgecolor='white', linewidth=0.8,
                yerr=[xerr_lo, xerr_hi], capsize=2.5, error_kw={'elinewidth': 1.0, 'alpha': 0.6}
            )

            if panel_idx == 0 and len(rects) > 0:
                handles.append(rects[0])
                labels.append(metric)
            
            for j, rect in enumerate(rects):
                height = vals[j]
                if not np.isnan(height):
                    text_y_offset = 4 + (m_idx % 2) * 12
                    ax.annotate(
                        f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, hi[j] if not np.isnan(hi[j]) else height),
                        xytext=(0, text_y_offset), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontfamily='monospace', rotation=0
                    )

        ax.set_title(f"Dataset: {dataset}", fontsize=15, fontweight='bold', pad=15)
        ax.set_ylabel("Metric Score", fontsize=12, fontweight='bold')
<<<<<<< HEAD
        ax.set_ylim(y_floor, y_ceiling)
=======
>>>>>>> a55a5e4e (fix: y lim in plots)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_labels, fontsize=10, fontweight='bold', rotation=30, ha='right')

    if all_hi:
        data_min, data_max = min(all_lo), max(all_hi)
        margin = 0.08 * (data_max - data_min) if data_max > data_min else 0.05
        axes[0, 0].set_ylim(data_min - margin, data_max + margin)

    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=len(labels), fontsize=11)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved Dataset-Grouped Bar Chart to {output_path}")

@with_cli_args(["+postprocessing=plots/grouped_bar_chart"])
@hydra.main(config_path="../../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    if config.get("mlflow_tracking_uri"):
        import mlflow
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        
    inputs = OmegaConf.to_object(config.get("inputs", []))
    metrics = OmegaConf.to_object(config.get("metrics", ["AUROC", "AUPRC"]))
    
    modality_order_cfg = config.get("modality_order", [])
    modality_order = OmegaConf.to_object(modality_order_cfg) if OmegaConf.is_config(modality_order_cfg) else modality_order_cfg
    
    condition_order_cfg = config.get("condition_order", [])
    condition_order = OmegaConf.to_object(condition_order_cfg) if OmegaConf.is_config(condition_order_cfg) else condition_order_cfg
    
    if not inputs:
        raise ValueError("No inputs provided.")
        
    df = load_plot_data(inputs)
    if df.empty:
        return
        
    with TemporaryDirectory() as output_dir:
        out_path = Path(output_dir) / "bars_dataset_grouped.png"
        create_dataset_grouped_bar_chart(df, metrics, modality_order, condition_order, out_path)
        logger.log_artifacts(
            local_dir=str(output_dir), 
            artifact_path=config.get("mlflow_artifact_path", "plots")
        )

if __name__ == "__main__":
    main()