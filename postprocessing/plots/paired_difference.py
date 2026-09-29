"""Paired comparison of two models, per dataset and metric, from nuclei predictions.

Layout: one row per dataset, one column group per metric. Each cell has:

- a score panel: both models as dots with slide-level bootstrap 95% CIs, plus
  faint vertical reference lines (e.g. a no-context model and the best full
  model) and, for AUPRC, the chance level (= positive nuclei rate);
- a difference panel: candidate - baseline with a *paired* slide-level
  bootstrap 95% CI (both models are scored on the same resampled slides in
  every replicate) and a vertical line at 0.

Metrics are computed from the per-nucleus prediction parquets rather than the
logged summary metrics, because the paired CI needs both models' predictions
on the same nuclei. Labels are built exactly as in `postprocessing/crop/metrics.py`.
"""

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.lines import Line2D
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from tqdm import tqdm

from postprocessing.crop.metrics import get_predictions


matplotlib.rcParams.update(
    {
        "figure.max_open_warning": 0,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    }
)

BASELINE_COLOR = "#2a78d6"
CANDIDATE_COLOR = "#eb6834"
DIFF_COLOR = "#0b0b0b"
REFERENCE_COLOR = "#8a8984"
TEXT_SECONDARY = "#52514e"
REFERENCE_STYLES = ["--", ":", (0, (8, 3))]
CHANCE_STYLE = "-."
LABEL_BOX = {"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.85}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------


def load_dataset(ds_cfg: dict, labels: list[str]) -> pd.DataFrame:
    """Returns one row per nucleus: slide_id, id, label, and one column per model.

    Only nuclei predicted by every model are kept, so all metrics (and the
    paired differences) are computed on exactly the same set of nuclei.
    """
    label_col = ds_cfg["label_column"]
    metadata_df = pd.read_parquet(download_artifacts(ds_cfg["metadata_uri"]))

    merged: pd.DataFrame | None = None
    for label in labels:
        print(f"Fetching: {ds_cfg['name']} | {label}")
        preds_dir = Path(download_artifacts(ds_cfg["models"][label]))
        preds = get_predictions(metadata_df["slide_id"], preds_dir)
        preds = preds[["slide_id", "id", "nuclei_prediction"]].rename(
            columns={"nuclei_prediction": label}
        )
        if merged is None:
            merged = preds
            continue
        n_before = len(merged)
        merged = merged.merge(preds, on=["slide_id", "id"], how="inner")
        if len(merged) != n_before or len(merged) != len(preds):
            print(
                f"  -> WARNING: nuclei differ between models; kept the "
                f"{len(merged)} nuclei common to all (was {n_before}, "
                f"'{label}' has {len(preds)})",
                file=sys.stderr,
            )
    assert merged is not None

    supervision_df = pd.read_parquet(ds_cfg["supervision_dir"])
    merged = merged.merge(
        supervision_df[["slide_id", "id", label_col]],
        on=["slide_id", "id"],
        how="left",
    )
    merged = merged.merge(
        metadata_df[["slide_id", "is_carcinoma"]], on="slide_id", how="left"
    )
    merged["label"] = merged[label_col].fillna(0).astype(int)
    merged.loc[~merged["is_carcinoma"].astype(bool), "label"] = 0
    return merged


# --------------------------------------------------------------------------
# Weighted metrics (slide-level bootstrap == integer weights per slide)
# --------------------------------------------------------------------------


class RankedScores:
    """One model's scores sorted once, so each bootstrap replicate is O(N).

    Resampling slides with replacement is equivalent to weighting every
    nucleus by its slide's draw count, so AUROC/AUPRC of a replicate can be
    computed from weighted cumulative sums over the pre-sorted scores. Ties
    are grouped into one threshold, matching sklearn.
    """

    def __init__(self, scores: np.ndarray, y: np.ndarray, slide_idx: np.ndarray):
        order = np.argsort(-scores, kind="mergesort")
        s_sorted = scores[order]
        self.y = y[order].astype(np.float64)
        self.slide_idx = slide_idx[order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(s_sorted)) + 1]

    def metrics(self, slide_weights: np.ndarray) -> tuple[float, float]:
        w = slide_weights[self.slide_idx]
        tp = np.add.reduceat(w * self.y, self.starts)
        fp = np.add.reduceat(w, self.starts) - tp
        cum_tp, cum_fp = np.cumsum(tp), np.cumsum(fp)
        n_pos, n_neg = cum_tp[-1], cum_fp[-1]
        if n_pos <= 0 or n_neg <= 0:
            return np.nan, np.nan

        tpr = np.r_[0.0, cum_tp / n_pos]
        fpr = np.r_[0.0, cum_fp / n_neg]
        auroc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))

        denom = cum_tp + cum_fp
        precision = np.divide(cum_tp, denom, out=np.zeros_like(cum_tp), where=denom > 0)
        auprc = float(np.sum(np.diff(tpr) * precision))
        return auroc, auprc


def evaluate_dataset(
    df: pd.DataFrame,
    baseline: str,
    candidate: str,
    references: list[str],
    n_bootstrap: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Point estimates for all models; marginal and paired CIs for the compared pair."""
    slide_idx, slides = pd.factorize(df["slide_id"])
    y = df["label"].to_numpy()
    n_slides = len(slides)

    ranked = {
        m: RankedScores(df[m].to_numpy(dtype=np.float64), y, slide_idx)
        for m in [baseline, candidate, *references]
    }
    ones = np.ones(n_slides)
    point = {
        m: dict(zip(("AUROC", "AUPRC"), r.metrics(ones), strict=True))
        for m, r in ranked.items()
    }

    boot = {m: [] for m in (baseline, candidate)}
    for _ in tqdm(range(n_bootstrap), desc="Paired bootstrap"):
        weights = np.bincount(
            rng.integers(0, n_slides, n_slides), minlength=n_slides
        ).astype(np.float64)
        for m, samples in boot.items():
            samples.append(ranked[m].metrics(weights))

    b_base = np.asarray(boot[baseline])
    b_cand = np.asarray(boot[candidate])
    valid = ~np.isnan(b_base).any(axis=1) & ~np.isnan(b_cand).any(axis=1)
    b_base, b_cand = b_base[valid], b_cand[valid]
    b_diff = b_cand - b_base

    rows = []
    for j, metric in enumerate(("AUROC", "AUPRC")):
        for role, m, samples in (
            ("baseline", baseline, b_base[:, j]),
            ("candidate", candidate, b_cand[:, j]),
            ("difference", f"{candidate} - {baseline}", b_diff[:, j]),
        ):
            value = (
                point[candidate][metric] - point[baseline][metric]
                if role == "difference"
                else point[m][metric]
            )
            lo, hi = np.percentile(samples, [2.5, 97.5])
            rows.append(
                {
                    "metric": metric,
                    "role": role,
                    "model": m,
                    "value": value,
                    "lo": lo,
                    "hi": hi,
                }
            )
        for ref in references:
            rows.append(
                {
                    "metric": metric,
                    "role": "reference",
                    "model": ref,
                    "value": point[ref][metric],
                    "lo": np.nan,
                    "hi": np.nan,
                }
            )
        if metric == "AUPRC":
            rows.append(
                {
                    "metric": metric,
                    "role": "chance",
                    "model": "positive rate",
                    "value": float(y.mean()),
                    "lo": np.nan,
                    "hi": np.nan,
                }
            )

    out = pd.DataFrame(rows)
    out["n_nuclei"] = len(df)
    out["n_slides"] = n_slides
    out["n_bootstrap_valid"] = int(valid.sum())
    return out


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------


def _padded(lo: float, hi: float, frac: float = 0.12) -> tuple[float, float]:
    pad = (hi - lo) * frac if hi > lo else 0.02
    return lo - pad, hi + pad


def create_plot(
    results: pd.DataFrame,
    datasets: list[str],
    metrics: list[str],
    baseline: str,
    candidate: str,
    references: list[str],
    output_path: Path,
) -> None:
    n_rows, n_metrics = len(datasets), len(metrics)
    width_ratios: list[float] = []
    for i in range(n_metrics):
        width_ratios += [3.0, 2.0] + ([0.35] if i < n_metrics - 1 else [])

    fig = plt.figure(figsize=(5.2 * n_metrics, 2.3 * n_rows + 0.9))
    gs = fig.add_gridspec(
        n_rows, len(width_ratios), width_ratios=width_ratios, wspace=0.14, hspace=0.55
    )
    ref_styles = {
        r: REFERENCE_STYLES[i % len(REFERENCE_STYLES)] for i, r in enumerate(references)
    }
    y_pos = {baseline: 0, candidate: 1}
    colors = {baseline: BASELINE_COLOR, candidate: CANDIDATE_COLOR}

    for m_idx, metric in enumerate(metrics):
        res_m = results[results["metric"] == metric]

        # Shared x-range per metric across datasets, so rows are comparable by eye.
        scores = res_m[res_m["role"].isin(["baseline", "candidate", "reference"])]
        s_lim = _padded(
            np.nanmin(scores[["value", "lo"]].to_numpy()),
            np.nanmax(scores[["value", "hi"]].to_numpy()),
        )
        diffs = res_m[res_m["role"] == "difference"]
        d_abs = max(np.nanmax(np.abs(diffs[["lo", "hi"]].to_numpy())), 1e-3)
        d_lim = (-d_abs * 1.25, d_abs * 1.25)

        col = m_idx * 3
        for r_idx, dataset in enumerate(datasets):
            res = res_m[res_m["dataset"] == dataset].set_index("role")
            ax_s = fig.add_subplot(gs[r_idx, col])
            ax_d = fig.add_subplot(gs[r_idx, col + 1])

            # --- score panel -------------------------------------------------
            refs = res_m[(res_m["dataset"] == dataset) & (res_m["role"] == "reference")]
            for _, ref in refs.iterrows():
                ax_s.axvline(
                    ref["value"],
                    color=REFERENCE_COLOR,
                    linestyle=ref_styles[ref["model"]],
                    linewidth=1.2,
                    zorder=1,
                )
            if "chance" in res.index:
                chance = float(res.loc["chance", "value"])
                on_axis = s_lim[0] <= chance <= s_lim[1]
                if on_axis:
                    ax_s.axvline(
                        chance,
                        color=REFERENCE_COLOR,
                        linestyle=CHANCE_STYLE,
                        linewidth=1.2,
                        zorder=1,
                    )
                ax_s.text(
                    0.02,
                    0.03,
                    f"chance = {chance:.3f}" + ("" if on_axis else " (off axis)"),
                    transform=ax_s.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=8,
                    color=TEXT_SECONDARY,
                    bbox=LABEL_BOX,
                    zorder=4,
                )

            for role, model in (("baseline", baseline), ("candidate", candidate)):
                row = res.loc[role]
                ax_s.errorbar(
                    row["value"],
                    y_pos[model],
                    xerr=[[row["value"] - row["lo"]], [row["hi"] - row["value"]]],
                    fmt="o",
                    color=colors[model],
                    ecolor=colors[model],
                    markersize=7,
                    linewidth=2,
                    capsize=0,
                    zorder=3,
                )
                ax_s.text(
                    row["value"],
                    y_pos[model] - 0.22,
                    f"{row['value']:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color=TEXT_SECONDARY,
                    bbox=LABEL_BOX,
                    zorder=4,
                )

            ax_s.set_xlim(s_lim)
            ax_s.set_ylim(1.6, -0.6)
            ax_s.set_yticks([0, 1])
            if m_idx == 0:
                ax_s.set_yticklabels([baseline, candidate], fontsize=10)
            else:
                ax_s.set_yticklabels([])
            ax_s.set_title(
                f"{dataset} · {metric}", fontsize=11, fontweight="bold", loc="left"
            )

            # --- difference panel --------------------------------------------
            row = res.loc["difference"]
            d_val, d_lo, d_hi = (
                np.round(row[k], 3) + 0.0 for k in ("value", "lo", "hi")
            )
            ax_d.axvline(0, color=TEXT_SECONDARY, linewidth=1.0, zorder=1)
            ax_d.errorbar(
                row["value"],
                0,
                xerr=[[row["value"] - row["lo"]], [row["hi"] - row["value"]]],
                fmt="D",
                color=DIFF_COLOR,
                ecolor=DIFF_COLOR,
                markersize=6,
                linewidth=2,
                capsize=0,
                zorder=3,
            )
            ax_d.text(
                0.5,
                0.92,
                f"{d_val:+.3f} [{d_lo:+.3f}, {d_hi:+.3f}]",
                transform=ax_d.transAxes,
                ha="center",
                va="top",
                fontsize=8,
                color=TEXT_SECONDARY,
                fontfamily="monospace",
            )
            ax_d.set_xlim(d_lim)
            ax_d.set_ylim(-1, 1)
            ax_d.set_yticks([])
            ax_d.set_title("Δ paired", fontsize=9, color=TEXT_SECONDARY)

            for ax, fmt, nbins in ((ax_s, "%.2f", 4), (ax_d, "%+.2f", 3)):
                ax.grid(axis="x", color="#e1e0d9", linewidth=0.8)
                ax.set_axisbelow(True)
                ax.spines[["top", "right", "left"]].set_visible(False)
                ax.tick_params(axis="y", length=0)
                ax.tick_params(axis="x", labelsize=8, colors=TEXT_SECONDARY)
                ax.xaxis.set_major_locator(
                    ticker.MaxNLocator(nbins=nbins, symmetric=ax is ax_d)
                )
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter(fmt))

            if r_idx == n_rows - 1:
                ax_s.set_xlabel(f"{metric} (95% CI)", fontsize=9)
                ax_d.set_xlabel(f"{candidate} \u2212 {baseline}", fontsize=9)

    handles = [
        Line2D([], [], color=BASELINE_COLOR, marker="o", linewidth=2, label=baseline),
        Line2D([], [], color=CANDIDATE_COLOR, marker="o", linewidth=2, label=candidate),
        Line2D(
            [],
            [],
            color=DIFF_COLOR,
            marker="D",
            linewidth=2,
            label="Paired difference (slide bootstrap)",
        ),
        *[
            Line2D(
                [],
                [],
                color=REFERENCE_COLOR,
                linestyle=ref_styles[r],
                linewidth=1.2,
                label=r,
            )
            for r in references
        ],
    ]
    if "AUPRC" in metrics:
        handles.append(
            Line2D(
                [],
                [],
                color=REFERENCE_COLOR,
                linestyle=CHANCE_STYLE,
                linewidth=1.2,
                label="AUPRC chance (positive nuclei rate)",
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 3),
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02 - 0.035 * (len(handles) > 3)),
    )

    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved paired difference plot to {output_path}")


@with_cli_args(["+postprocessing=plots/paired_difference"])
@hydra.main(
    config_path="../../configs", config_name="postprocessing", version_base=None
)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    if config.get("mlflow_tracking_uri"):
        import mlflow

        mlflow.set_tracking_uri(config.mlflow_tracking_uri)

    baseline = config.comparison.baseline
    candidate = config.comparison.candidate
    metrics = list(config.get("metrics", ["AUROC", "AUPRC"]))
    datasets_cfg = OmegaConf.to_container(config.datasets, resolve=True)
    rng = np.random.default_rng(config.get("seed", 0))

    all_results, dataset_names, used_refs = [], [], []
    for ds in datasets_cfg:
        models = {k: v for k, v in ds["models"].items() if v}
        for required in (baseline, candidate):
            if required not in models:
                raise ValueError(f"{ds['name']}: no predictions URI for '{required}'")
        refs = [r for r in config.get("references", []) if r in models]
        for r in config.get("references", []):
            if r not in models:
                print(
                    f"{ds['name']}: skipping reference '{r}' (no URI)", file=sys.stderr
                )

        df = load_dataset({**ds, "models": models}, [baseline, candidate, *refs])
        res = evaluate_dataset(
            df, baseline, candidate, refs, config.get("n_bootstrap", 2000), rng
        )
        res.insert(0, "dataset", ds["name"])
        all_results.append(res)
        dataset_names.append(ds["name"])
        used_refs += [r for r in refs if r not in used_refs]

    results = pd.concat(all_results, ignore_index=True)
    print(results.to_string(index=False))

    with TemporaryDirectory() as output_dir:
        out = Path(output_dir)
        results.to_csv(out / "paired_difference.csv", index=False)
        create_plot(
            results,
            dataset_names,
            metrics,
            baseline,
            candidate,
            used_refs,
            out / "paired_difference.png",
        )
        logger.log_artifacts(
            local_dir=str(out),
            artifact_path=config.get("mlflow_artifact_path", "plots"),
        )


if __name__ == "__main__":
    main()
