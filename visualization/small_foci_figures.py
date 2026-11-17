"""Figures of postprocessing.small_foci_radboud, thesis style; no slide ids.

- ``recall_fpr_<split>``: all models in one figure. Left: recall at each seed's Youden threshold per
  closed-focus-area bin (pooled over the bin's carcinoma nuclei; seed mean, whiskers = seed range;
  the two smallest bins shaded). Right: FPR of benign nuclei of positive slides by distance outside
  the nearest focus.
- ``weight_curve_validation``: step 0, seed-mean validation AUPRC and small-focus recall against the
  weight of the main model in the fused logit.
"""

from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from postprocessing.bootstrap_sources import THESIS_RC


FIG_W_CM = 16.0
INK, MUTED, GRID = "#1f1f1f", "#8a8a8a", "#e6e6e6"
# fixed colours: references grey/black, interventions a qualitative set; B1 components dashed
COLOURS = {
    "main": "#1f1f1f",
    "self": "#8a8a8a",
    "control": "#2C6E9B",
    "fusion_mean": "#B5651D",
    "fusion_w": "#5E8C61",
    "a1_focus_seeding": "#B5651D",
    "a2_focus_weight": "#5E8C61",
    "a3_crop_balanced": "#8E5A9B",
    "b1_self_head": "#C0392B",
    "b2_k8": "#D4A017",
    "b2_k12": "#17A2B8",
}
LABELS = {
    "main": "Main (thesis)",
    "self": "Self-only",
    "control": "Control",
    "fusion_mean": "Fusion, w = 0.5",
    "fusion_w": "Fusion, selected w",
    "a1_focus_seeding": "A1 focus seeding",
    "a2_focus_weight": "A2 focus-size weight",
    "a3_crop_balanced": "A3 crop-balanced",
    "b1_self_head": "B1 self head",
    "b2_k8": "B2 k = 8",
    "b2_k12": "B2 k = 12",
}


def style(model: str) -> dict:
    base, _, component = model.partition(":")
    return {
        "color": COLOURS.get(base, MUTED),
        "linestyle": {"": "-", "context": "--", "self": ":"}[component],
        "label": LABELS.get(base, base) + (f" ({component})" if component else ""),
    }


def save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def recall_fpr_figure(
    per_seed: pd.DataFrame, bins: list[str], bands: list[str], stem: Path, title: str
) -> None:
    rec_cols = [f"recall_bin_{b}" for b in bins]
    fpr_cols = [f"FPR_out_{b}um" for b in bands]
    present = [i for i, c in enumerate(rec_cols) if per_seed[c].notna().any()]
    with plt.rc_context(THESIS_RC):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(FIG_W_CM / 2.54, 6.2 / 2.54), width_ratios=[2.2, 1]
        )
        x1 = np.arange(len(present))
        ax1.axvspan(-0.5, 1.5, color=GRID, zorder=0)
        for model, g in per_seed.groupby("model", sort=False):
            st = style(str(model))
            for ax, cols, x in (
                (ax1, [rec_cols[i] for i in present], x1),
                (ax2, fpr_cols, np.arange(len(bands))),
            ):
                v = g[cols].to_numpy(float)
                mean = np.nanmean(v, axis=0)
                ax.plot(x, mean, marker="o", ms=2.5, lw=1.0, **st)
                if len(v) > 1:
                    ax.vlines(
                        x,
                        np.nanmin(v, 0),
                        np.nanmax(v, 0),
                        color=st["color"],
                        lw=0.6,
                        alpha=0.7,
                    )
        ax1.set_xticks(x1, [bins[i] for i in present], rotation=45, ha="right")
        ax1.set_xlabel(r"Focus area (closed, mm$^2$)")
        ax1.set_ylabel("Recall (Youden threshold)")
        ax2.set_xticks(np.arange(len(bands)), bands)
        ax2.set_xlabel(r"Distance outside focus ($\mu$m)")
        ax2.set_ylabel("FPR, benign nuclei")
        for ax in (ax1, ax2):
            ax.grid(axis="y", color=GRID, lw=0.5)
            ax.spines[["top", "right"]].set_visible(False)
        ax1.legend(frameon=False, fontsize=6, ncol=2, loc="lower right")
        fig.suptitle(title, fontsize=8)
        fig.tight_layout()
        save(fig, stem)


def fusion_figure(curve: pd.DataFrame, best: float, stem: Path) -> None:
    with plt.rc_context(THESIS_RC):
        fig, ax = plt.subplots(figsize=(FIG_W_CM / 2 / 2.54, 5.5 / 2.54))
        ax.plot(curve["w_main"], curve["AUPRC"], color=INK, lw=1.0, label="AUPRC")
        ax.set_xlabel("Weight of the main model's logit")
        ax.set_ylabel("Validation AUPRC")
        ax2 = ax.twinx()
        ax2.plot(
            curve["w_main"],
            curve["recall_small"],
            color=COLOURS["fusion_mean"],
            lw=1.0,
            label="Recall, two smallest bins",
        )
        ax2.set_ylabel("Recall, two smallest bins", color=COLOURS["fusion_mean"])
        ax.axvline(best, color=MUTED, lw=0.6, ls="--")
        ax.spines[["top"]].set_visible(False)
        fig.tight_layout()
        save(fig, stem)
