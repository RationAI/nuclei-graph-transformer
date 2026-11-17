# ruff: noqa: RUF002  (µ, ², minus signs in text are intended)
"""Figures of postprocessing.focus_size_analysis_radboud (D1–D3), thesis style; no slide ids.

Hue = information source (thesis colours: shape blue, texture orange, blank token green), line style
= context (k-NN solid, self-only dashed, dense dotted). Score = seed-mean probability; recall / FPR
at each model's Youden threshold.

- ``d1_focus_size_c<d>``: median of the focus median scores and pooled recall per focus-area bin.
- ``d1_focus_scatter_c<primary>``: per model, focus median score vs area, binned median, threshold.
- ``d2_profile_<focus|gland>``: per 10 µm signed-distance bin, median score (carcinoma nuclei
  inside, non-carcinoma outside) and the share at or above the threshold (recall inside, FPR
  outside); all foci, small and large foci.
- ``d3_patterns``: recall per pattern, within-slide AUROC per pattern, recall vs area within G3- and
  G4-dominant foci.

Redraw from a results folder: ``uv run -m visualization.focus_size_figures --results-dir DIR``.
"""

from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from postprocessing.bootstrap_sources import THESIS_RC


TEX_PT_PER_IN = 72.27
TEXT_W_IN = 455 / TEX_PT_PER_IN
RC = {**THESIS_RC, "font.size": 8}
INK, MUTED, GRID = "#1f1f1f", "#8a8a8a", "#e6e6e6"
SHAPE, TEXTURE, POSITION = "#2C6E9B", "#B5651D", "#5E8C61"
STYLE = {
    "shape_knn": (SHAPE, "-"),
    "shape_self": (SHAPE, "--"),
    "shape_dense": (SHAPE, ":"),
    "blank_knn": (POSITION, "-"),
    "texture_self": (TEXTURE, "--"),
    "texture_knn": (TEXTURE, "-"),
}
MARKERS = {
    "shape_knn": "o",
    "shape_self": "s",
    "shape_dense": "D",
    "blank_knn": "^",
    "texture_self": "v",
    "texture_knn": "P",
}


def fmt(v: float) -> str:
    """Signed number with a math minus (cmr10 has no U+2212)."""
    return f"${v:+.2f}$"


def save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def style_ax(ax: plt.Axes) -> None:
    ax.grid(color=GRID, lw=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK)


def legend(fig: plt.Figure, ax: plt.Axes, ncol: int = 6) -> None:
    h, lab = ax.get_legend_handles_labels()
    fig.legend(
        h,
        lab,
        loc="lower center",
        ncol=ncol,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )


def d1_lines(
    config: DictConfig, bins: pd.DataFrame, closing: int, out: Path, subset: str = "all"
) -> None:
    models = list(config.models)
    lab = {k: v.short for k, v in config.models.items()}
    b = bins[(bins.closing_um == closing) & (bins.foci == subset)]
    with plt.rc_context(RC):
        fig, axs = plt.subplots(1, 2, figsize=(TEXT_W_IN, 2.5))
        for m in models:
            d = b[b.model == m]
            d = d[d.n_foci >= 5]
            c, ls = STYLE[m]
            kw = {
                "color": c,
                "ls": ls,
                "lw": 1.2,
                "marker": MARKERS[m],
                "ms": 3,
                "label": lab[m],
            }
            axs[0].plot(d.median_area_mm2, d.median_score_median, **kw)
            axs[1].plot(d.median_area_mm2, d.recall_pooled, **kw)
        for ax, yl in zip(
            axs, ("Median score of the focus", "Recall (pooled)"), strict=True
        ):
            ax.set_xscale("log")
            ax.set_xlabel(r"Focus area (mm$^2$)")
            ax.set_ylabel(yl)
            ax.set_ylim(0, 1)
            style_ax(ax)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        legend(fig, axs[0])
        save(
            fig,
            out,
            f"d1_focus_size_c{closing}" + ("" if subset == "all" else f"_{subset}"),
        )


def d1_scatter(
    config: DictConfig,
    per_focus: dict,
    bins: pd.DataFrame,
    corr: pd.DataFrame,
    closing: int,
    out: Path,
    thresholds: dict[str, float],
) -> None:
    models = list(config.models)
    a = config.analysis
    with plt.rc_context(RC):
        fig, axs = plt.subplots(
            2, 3, figsize=(TEXT_W_IN, 4.2), sharex=True, sharey=True
        )
        for ax in axs.ravel()[len(models) :]:
            ax.set_visible(False)
        for ax, m in zip(axs.ravel(), models, strict=False):
            fs = per_focus[(m, closing)]
            fs = fs[fs.n_carcinoma >= int(a.min_focus_nuclei)]
            ax.scatter(
                fs.area_mm2,
                fs.median_score,
                s=1.5,
                color=MUTED,
                alpha=0.25,
                lw=0,
                rasterized=True,
            )
            d = bins[
                (bins.model == m)
                & (bins.closing_um == closing)
                & (bins.foci == "all")
                & (bins.n_foci >= 5)
            ]
            c, ls = STYLE[m]
            ax.plot(
                d.median_area_mm2,
                d.median_score_median,
                color=c,
                ls=ls,
                lw=1.4,
                marker=MARKERS[m],
                ms=3,
            )
            ax.fill_between(
                d.median_area_mm2,
                d.median_score_q1,
                d.median_score_q3,
                color=c,
                alpha=0.15,
                lw=0,
            )
            if m in thresholds:
                ax.axhline(thresholds[m], color=INK, lw=0.6, ls=(0, (2, 2)))
            r = corr[
                (corr.model == m)
                & (corr.closing_um == closing)
                & (corr.foci == "all")
                & (corr.variable == "median_score")
            ].iloc[0]
            ax.set_title(
                f"{config.models[m].short}\n$\\rho$ = {fmt(r.spearman_rho)} [{fmt(r.ci_lo)}, {fmt(r.ci_hi)}]",
                fontsize=8,
            )
            ax.set_xscale("log")
            ax.set_ylim(0, 1)
            style_ax(ax)
        for ax in axs[1]:
            ax.set_xlabel(r"Focus area (mm$^2$)")
        for ax in axs[:, 0]:
            ax.set_ylabel("Focus median score")
        fig.tight_layout()
        save(fig, out, f"d1_focus_scatter_c{closing}")


def d2_profile(
    config: DictConfig, prof: pd.DataFrame, boundary: str, out: Path
) -> None:
    models = list(config.models)
    lab = {k: v.short for k, v in config.models.items()}
    a = config.analysis
    titles = {
        "all": "All foci",
        "small": rf"Foci $<$ {a.small_focus_mm2:g} mm$^2$",
        "large": rf"Foci $>$ {a.large_focus_mm2:g} mm$^2$",
    }
    p = prof[(prof.boundary == boundary) & prof.bin_mid_um.notna()]
    with plt.rc_context(RC):
        fig, axs = plt.subplots(
            2, 3, figsize=(TEXT_W_IN, 4.3), sharex=True, sharey="row"
        )
        for j, restriction in enumerate(("all", "small", "large")):
            for m in models:
                d = p[(p.foci == restriction) & (p.model == m)].sort_values(
                    "bin_mid_um"
                )
                inside = d.bin_mid_um > 0
                score = np.where(
                    inside, d.median_score_carcinoma, d.median_score_noncarcinoma
                )
                share = np.where(inside, d.recall, d.FPR)
                n = np.where(inside, d.n_carcinoma, d.n_noncarcinoma)
                score = np.where(n >= 20, score, np.nan)
                share = np.where(n >= 20, share, np.nan)
                c, ls = STYLE[m]
                kw = {"color": c, "ls": ls, "lw": 1.1, "label": lab[m]}
                for sel in (
                    ~inside.to_numpy(),
                    inside.to_numpy(),
                ):  # no line across the boundary
                    axs[0, j].plot(d.bin_mid_um[sel], score[sel], **kw)
                    axs[1, j].plot(d.bin_mid_um[sel], share[sel], **kw)
                    kw.pop("label", None)
            axs[0, j].set_title(titles[restriction], fontsize=8)
            for ax in axs[:, j]:
                ax.axvline(0, color=INK, lw=0.6)
                ax.set_ylim(0, 1)
                style_ax(ax)
            axs[1, j].set_xlabel(r"Signed distance to the boundary ($\mu$m)")
        axs[0, 0].set_ylabel("Median score")
        axs[1, 0].set_ylabel(r"FPR ($-$) / recall ($+$)")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        legend(fig, axs[0, 0])
        save(fig, out, f"d2_profile_{boundary}")


def d3_patterns(
    config: DictConfig,
    rates: pd.DataFrame,
    auc: pd.DataFrame,
    bins: pd.DataFrame,
    out: Path,
) -> None:
    models = list(config.models)
    lab = {k: v.short for k, v in config.models.items()}
    primary = int(config.geometry.primary_closing_um)
    pats = ("G3", "G4", "G5")
    off = np.linspace(-0.3, 0.3, len(models))
    with plt.rc_context(RC):
        fig, axs = plt.subplots(2, 2, figsize=(TEXT_W_IN, 4.6))
        ax = axs[0, 0]
        for o, m in zip(off, models, strict=True):
            d = (
                rates[(rates.model == m) & rates["class"].isin(pats)]
                .set_index("class")
                .loc[list(pats)]
            )
            c, _ = STYLE[m]
            x = np.arange(3) + o
            ax.errorbar(
                x,
                d.value,
                yerr=[d.value - d.ci_lo, d.ci_hi - d.value],
                fmt=MARKERS[m],
                color=c,
                ms=3.5,
                lw=0.8,
                label=lab[m],
            )
        ax.set_xticks(range(3), pats)
        ax.set_ylabel("Recall")
        ax.set_ylim(0, 1)
        ax.set_title("(a) Recall per pattern", fontsize=8, loc="left")
        style_ax(ax)
        ax = axs[0, 1]
        a = auc[auc.reference == "non-carcinoma"]
        for o, m in zip(off, models, strict=True):
            d = (
                a[(a.model == m) & a.pattern.isin(pats)]
                .set_index("pattern")
                .reindex(list(pats))
            )
            c, _ = STYLE[m]
            x = np.arange(3) + o
            ax.errorbar(
                x,
                d.median_AUROC,
                yerr=[d.median_AUROC - d.q1, d.q3 - d.median_AUROC],
                fmt=MARKERS[m],
                color=c,
                ms=3.5,
                lw=0.8,
            )
        ax.set_xticks(range(3), pats)
        ax.set_ylabel("Within-slide AUROC (median, IQR)")
        ax.set_ylim(0.5, 1)
        ax.set_title(
            "(b) Pattern vs the slide's non-carcinoma nuclei", fontsize=8, loc="left"
        )
        style_ax(ax)
        for ax, p, letter in ((axs[1, 0], 3, "c"), (axs[1, 1], 4, "d")):
            b = bins[
                (bins.closing_um == primary)
                & (bins.foci == f"G{p}-dominant")
                & (bins.n_foci >= 5)
            ]
            for m in models:
                d = b[b.model == m]
                c, ls = STYLE[m]
                ax.plot(
                    d.median_area_mm2,
                    d.recall_pooled,
                    color=c,
                    ls=ls,
                    lw=1.1,
                    marker=MARKERS[m],
                    ms=3,
                )
            ax.set_xscale("log")
            ax.set_ylim(0, 1)
            ax.set_xlabel(r"Focus area (mm$^2$)")
            ax.set_ylabel("Recall (pooled)")
            ax.set_title(f"({letter}) G{p}-dominant foci", fontsize=8, loc="left")
            style_ax(ax)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        legend(fig, axs[0, 0])
        save(fig, out, "d3_patterns")


def save_figures(
    config: DictConfig,
    tables: dict,
    per_focus: dict,
    out: Path,
    thresholds: dict[str, float],
) -> None:
    primary = int(config.geometry.primary_closing_um)
    for d in config.geometry.closings_um:
        d1_lines(config, tables["d1_bins"], int(d), out)
    d1_scatter(
        config,
        per_focus,
        tables["d1_bins"],
        tables["d1_spearman"],
        primary,
        out,
        thresholds,
    )
    for boundary in ("focus", "gland"):
        d2_profile(config, tables["d2_profile"], boundary, out)
    d3_patterns(config, tables["d3_rates"], tables["d3_auroc"], tables["d1_bins"], out)


def main() -> None:
    """Redraws the figures from the saved outputs of an analysis run (nothing is recomputed)."""
    import argparse
    import json

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    res = args.results_dir
    config = OmegaConf.load(res / "config-resolved.yaml")
    names = ("d1_spearman", "d1_bins", "d2_profile", "d3_rates", "d3_auroc")
    tables = {k: pd.read_csv(res / f"{k}.csv") for k in names}
    per_focus = {
        (m, int(d)): pd.read_csv(res / f"d1_foci_{m}_c{int(d)}.csv.gz")
        for m in config.models
        for d in config.geometry.closings_um
        if int(d) == int(config.geometry.primary_closing_um)
    }
    thresholds = {
        k: float(v["threshold"])
        for k, v in json.loads((res / "thresholds.json").read_text()).items()
    }
    save_figures(config, tables, per_focus, args.out_dir or res / "figures", thresholds)
    print(f"figures in {args.out_dir or res / 'figures'}")


if __name__ == "__main__":
    main()
