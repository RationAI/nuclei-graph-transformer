# ruff: noqa: RUF001, RUF002, RUF003, ISC004  (mathematical symbols in text are intended)
r"""Where two model configurations disagree: profiles of the disagreement sets (part A).

PANDA Radboud test set, all 14.7 M test nuclei. Stored outputs only: seed-mean probabilities
(three seeds each) of the with-context models of tab:res-sources (Q4: shape, texture and position
only, k-NN + RoVE) and of the context models of tab:res-context (Q3: self-only, k-NN and dense,
no positions in attention); the dominant stable SAE concept per nucleus saved by run 3c6e64a5
(`dominant_group.npy`); the epithelial mask of run 44ffd165 (carcinoma ∪ benign epithelium by
the vertex rule); the morphometry of the thesis concept analysis. No training, no inference.

Sets. At the nucleus threshold t = 0.40 (p ≥ t = carcinoma), a configuration is correct on a
nucleus if its prediction equals the label. For a pair (X, Y), "X > Y" = X correct and Y wrong,
"Y > X" the reverse; separately per true class (carcinoma, benign), and again restricted to
strong disagreements |p_X − p_Y| ≥ 0.5.

Per set (pair x direction x class x all/strong):
1. size and share of the class;
2. composition by dominant concept: share in the set, share in the class, enrichment ratio
   (set share / class share);
3. epithelial share in the set and in the class, and their difference;
4. morphometry: standardized difference of the set median from the class median,
   (median_set − median_class) / (IQR_class / 1.349), per feature; top 5 by |value|. OD only on
   the random per-slide OD sample. Point values, no CI.
CIs (1−3): percentile intervals from the shared stratified slide bootstrap of the thesis (seed
20261108, B = 2,000; resamples read, not redrawn); in each resample a share is a
multiplicity-weighted sum of per-slide counts over a weighted sum of per-slide totals. p (for
enrichment: log ratio vs 0; for the epithelial difference: vs 0) = 2 x the smaller tail
fraction, unadjusted (descriptive).

Outputs in `out_dir` (no slide ids): `sets.csv`, `concepts.csv`, `epithelium.csv`,
`morphometry.csv`, `tables.md`, `checks.json`, `figures/res_disagreement_{context,sources}.{pdf,png}`
(all disagreements; the strong ones are in the CSVs only).
"""

import json
import subprocess
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from nuclei_graph.mlflow_utils import tag_parent_run
from postprocessing.bootstrap_sources import (
    bootstrap_p,
    label_predictions,
    load_labels,
    load_run,
)
from postprocessing.context_concepts import xai_order
from postprocessing.results_context import REPO, epithelium_classes
from visualization.disagreement_figure import FIGURES, save_disagreement_figures


CLASSES = (("carcinoma", 1), ("benign", 0))
SUBSETS = ("all", "strong")
SOURCES = (
    "postprocessing/disagreement_analysis.py",
    "postprocessing/bootstrap_sources.py",
    "postprocessing/context_concepts.py",
    "postprocessing/results_context.py",
    "visualization/disagreement_figure.py",
    "configs/postprocessing/disagreement_analysis.yaml",
    "configs/postprocessing/results_context.yaml",
    "configs/postprocessing/results_sources.yaml",
)


def code_commit() -> str:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    dirty = git("status", "--porcelain", "--untracked-files=no", "--", *SOURCES)
    return f"{git('rev-parse', 'HEAD')}{' (dirty)' if dirty else ''}"


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def configurations(config: DictConfig) -> dict[str, list[str]]:
    """Eval runs of every configuration in a pair; Q4 runs checked against tab:res-sources."""
    reference = OmegaConf.load(REPO / config.sources_config).configurations
    runs = {}
    for name, c in config.q4_configurations.items():
        assert list(c.eval_runs) == list(reference[name].eval_runs), f"{name}: runs differ"
        runs[name] = list(c.eval_runs)
    for name in ("self", "knn", "dense"):
        runs[name] = list(config.configurations[name].eval_runs)
    for pair in config.pairs:
        assert pair.x in runs and pair.y in runs, pair.name
    assert all(len(r) == 3 for r in runs.values()), "three seeds each"
    return runs


def test_slides(config: DictConfig) -> pd.DataFrame:
    """Test slides sorted by id (the predictions' slide index), from the XAI slide table.

    The slide list must equal that of the saved resamples (which fixes the order); the
    carcinoma flags and nucleus counts are checked against the predictions and the grouping.
    """
    slides = pd.read_parquet(config.epithelium.slides)
    slides = slides[slides["split"] == "test"][["slide_id", "is_carcinoma"]]
    slides = slides.sort_values("slide_id", ignore_index=True)
    saved = np.load(config.resamples)
    assert saved["slide_ids"].tolist() == slides["slide_id"].tolist(), "slides differ"
    return slides


def load_inputs(config: DictConfig) -> dict:
    slides = test_slides(config)
    slide_ids = slides["slide_id"].tolist()
    carcinoma = slides["is_carcinoma"].to_numpy(bool)
    labels = load_labels(slides, Path(config.supervision_dir), config.label_column)

    reference, y, p_mean = None, None, {}
    for name, runs in configurations(config).items():
        total = None
        for run_id in runs:
            df, _ = load_run(
                run_id, slide_ids, Path(config.cache_dir), config.predictions_artifact
            )
            if reference is None:
                reference = df[["slide", "key"]].copy()
                y, missing = label_predictions(df, labels, carcinoma)
            assert np.array_equal(df["slide"].to_numpy(), reference["slide"].to_numpy())
            assert np.array_equal(df["key"].to_numpy(), reference["key"].to_numpy())
            p = df["p"].to_numpy(np.float64)
            total = p if total is None else total + p
        p_mean[name] = total / len(runs)
        print(f"{name}: loaded", flush=True)
    assert reference is not None and y is not None
    slide = reference["slide"].to_numpy().astype(np.int64)

    # dominant concept (run 3c6e64a5), counts asserted against its groups.csv
    cdir = Path(config.concepts.dir)
    group = np.load(cdir / "dominant_group.npy")
    group_names = json.loads((cdir / "checks.json").read_text())["groups"]
    assert len(group) == len(y)
    ref = pd.read_csv(cdir / "groups.csv").set_index("group")
    for g, name in enumerate(group_names):
        assert int((group == g).sum()) == ref.loc[name, "n_nuclei"], name
        assert int(((group == g) & (y == 1)).sum()) == ref.loc[name, "n_carcinoma"], name

    # epithelium (run 44ffd165): carcinoma ∪ benign epithelium, count asserted
    cls, _ = epithelium_classes(config, slides, slide, y)
    epithelial = (y == 1) | (cls == 2)
    ref_epi = pd.read_csv(config.epithelium_counts_reference).set_index("subset")
    assert int(epithelial.sum()) == ref_epi.loc["epithelium", "nuclei"], "epithelium"

    # morphometry, in the predictions' order
    order = xai_order(config, slide_ids, reference, y)
    m = config.morphometry
    poly_cols = [c for c in m.features if c not in ("irregularity", "knn8_mean_dist_um",
                                                    "pleomorphism_cv", "orientation_coherence",
                                                    "boundary_dist_um")]
    ctx_cols = [c for c in m.features if c not in poly_cols]
    poly = pd.read_parquet(m.polygons, columns=[*poly_cols, "od_random_sample"])
    ctx = pd.read_parquet(m.context, columns=ctx_cols)
    features = {}
    for c in poly_cols:
        v = poly[c].to_numpy(np.float64)
        if c in m.od_features:
            v = np.where(poly["od_random_sample"].to_numpy(bool), v, np.nan)
        features[c] = v[order]
    for c in ctx_cols:
        features[c] = ctx[c].to_numpy(np.float64)[order]

    return {
        "slide_ids": slide_ids,
        "carcinoma_slide": carcinoma,
        "slide": slide,
        "key": reference["key"].to_numpy(),
        "y": y,
        "p_mean": p_mean,
        "group": group.astype(np.int64),
        "group_names": group_names,
        "epithelial": epithelial,
        "features": features,
        "labels_missing": missing,
    }


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------


def disagreement_sets(config: DictConfig, data: dict) -> list[dict]:
    """One entry per pair x direction: masks of the sets (all classes together)."""
    t = float(config.threshold)
    y = data["y"]
    out = []
    for pair in config.pairs:
        px, py = data["p_mean"][pair.x], data["p_mean"][pair.y]
        cx, cy = (px >= t) == (y == 1), (py >= t) == (y == 1)
        strong = np.abs(px - py) >= config.strong_disagreement
        for first, second, mask in (
            (pair.x_short, pair.y_short, cx & ~cy),
            (pair.y_short, pair.x_short, cy & ~cx),
        ):
            out.append({
                "pair": pair.name,
                "question": pair.question,
                "direction": f"{first} > {second}",
                "forward": first == pair.x_short,
                "masks": {"all": mask, "strong": mask & strong},
            })
    return out


class Bootstrap:
    """Ratios of per-slide sums under the saved slide resamples."""

    def __init__(self, config: DictConfig, slide: NDArray, n_slides: int) -> None:
        saved = np.load(config.resamples)
        self.w = saved["multiplicities"].astype(np.float64)
        self.slide = slide
        self.n_slides = n_slides
        self.pct = (1 - config.bootstrap.ci_level) / 2 * 100

    def per_slide(self, mask: NDArray, cell: NDArray | None = None, n_cells: int = 1) -> NDArray:
        """Per-slide (x cell) counts of `mask`: n_slides x n_cells."""
        idx = self.slide[mask] * n_cells + (0 if cell is None else cell[mask])
        return np.bincount(idx, minlength=self.n_slides * n_cells).reshape(
            self.n_slides, n_cells
        ).astype(np.float64)

    def reps(self, counts: NDArray) -> NDArray:
        return self.w @ counts

    def ci(self, reps: NDArray) -> tuple[float, float]:
        lo, hi = np.nanpercentile(reps, [self.pct, 100 - self.pct])
        return float(lo), float(hi)


def profile(config: DictConfig, data: dict) -> tuple[pd.DataFrame, ...]:
    y, group, epi = data["y"], data["group"], data["epithelial"]
    names = data["group_names"]
    n_groups = len(names)
    boot = Bootstrap(config, data["slide"], len(data["slide_ids"]))
    sets = disagreement_sets(config, data)

    set_rows, concept_rows, epi_rows, morph_rows = [], [], [], []
    for cname, cval in CLASSES:
        in_class = y == cval
        class_g = boot.per_slide(in_class, group, n_groups)  # slides x groups
        class_n = class_g.sum(1, keepdims=True)
        class_epi = boot.per_slide(in_class & epi)
        rep_class_g, rep_class_n = boot.reps(class_g), boot.reps(class_n)
        rep_class_epi = boot.reps(class_epi)
        class_share_g = class_g.sum(0) / class_n.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            rep_class_share_g = rep_class_g / rep_class_n
            rep_class_epi_share = (rep_class_epi / rep_class_n)[:, 0]
        class_epi_share = class_epi.sum() / class_n.sum()

        # morphometry of the class: median and robust SD (IQR / 1.349), non-missing values
        class_stats = {}
        for f, v in data["features"].items():
            vals = v[in_class]
            vals = vals[~np.isnan(vals)]
            q25, q50, q75 = np.quantile(vals, [0.25, 0.5, 0.75])
            class_stats[f] = (float(q50), float((q75 - q25) / 1.349), len(vals))

        for s in sets:
            for subset in SUBSETS:
                mask = s["masks"][subset] & in_class
                key = {"pair": s["pair"], "question": s["question"],
                       "direction": s["direction"], "forward": s["forward"],
                       "true_class": cname, "subset": subset}
                set_g = boot.per_slide(mask, group, n_groups)
                set_n = set_g.sum(1, keepdims=True)
                n = int(set_n.sum())
                rep_n = boot.reps(set_n)[:, 0]
                with np.errstate(invalid="ignore", divide="ignore"):
                    rep_share = rep_n / rep_class_n[:, 0]
                set_rows.append(key | {
                    "n": n,
                    "n_class": int(class_n.sum()),
                    "share_of_class": n / float(class_n.sum()),
                    **dict(zip(("share_ci_lo", "share_ci_hi"), boot.ci(rep_share), strict=True)),
                    "n_slides": int((set_n[:, 0] > 0).sum()),
                })

                # 2. concept composition
                rep_set_g = boot.reps(set_g)
                with np.errstate(invalid="ignore", divide="ignore"):
                    rep_set_share = rep_set_g / rep_n[:, None]
                    rep_enrich = rep_set_share / rep_class_share_g
                for g, gname in enumerate(names):
                    share = set_g[:, g].sum() / n if n else np.nan
                    enrich = share / class_share_g[g] if class_share_g[g] > 0 else np.nan
                    r = rep_enrich[:, g]
                    log_r = np.log(r[np.isfinite(r) & (r > 0)])
                    concept_rows.append(key | {
                        "group": gname,
                        "n_in_set": int(set_g[:, g].sum()),
                        "share_set": share,
                        "share_class": float(class_share_g[g]),
                        "enrichment": enrich,
                        **dict(zip(("ci_lo", "ci_hi"), boot.ci(r), strict=True)),
                        "p": bootstrap_p(log_r) if len(log_r) else np.nan,
                        "resamples_undefined": int((~np.isfinite(r) | (r <= 0)).sum()),
                    })

                # 3. epithelial share
                set_epi = boot.per_slide(mask & epi)
                rep_set_epi = boot.reps(set_epi)[:, 0]
                with np.errstate(invalid="ignore", divide="ignore"):
                    rep_epi_share = rep_set_epi / rep_n
                diff = rep_epi_share - rep_class_epi_share
                ok = ~np.isnan(diff)
                epi_share = set_epi.sum() / n if n else np.nan
                epi_rows.append(key | {
                    "n": n,
                    "epithelial_share_set": epi_share,
                    **dict(zip(("set_ci_lo", "set_ci_hi"), boot.ci(rep_epi_share), strict=True)),
                    "epithelial_share_class": float(class_epi_share),
                    **dict(zip(("class_ci_lo", "class_ci_hi"),
                               boot.ci(rep_class_epi_share), strict=True)),
                    "difference": epi_share - class_epi_share,
                    **dict(zip(("diff_ci_lo", "diff_ci_hi"), boot.ci(diff), strict=True)),
                    "p": bootstrap_p(diff[ok]) if ok.any() and cval == 0 else np.nan,
                })

                # 4. morphometry
                rows = []
                for f, v in data["features"].items():
                    vals = v[mask]
                    vals = vals[~np.isnan(vals)]
                    med_c, sd_c, _ = class_stats[f]
                    med_s = float(np.median(vals)) if len(vals) else np.nan
                    rows.append(key | {
                        "feature": f,
                        "label": config.morphometry.features[f],
                        "n_measured": len(vals),
                        "median_set": med_s,
                        "median_class": med_c,
                        "robust_sd_class": sd_c,
                        "std_difference": (med_s - med_c) / sd_c,
                    })
                order = np.argsort([-abs(r["std_difference"]) if np.isfinite(r["std_difference"])
                                    else 0 for r in rows], kind="stable")
                for rank, i in enumerate(order, 1):
                    rows[i]["rank"] = rank
                    rows[i]["top"] = rank <= config.morphometry.top_n
                morph_rows += rows
        print(f"{cname}: profiled", flush=True)

    return (pd.DataFrame(set_rows), pd.DataFrame(concept_rows), pd.DataFrame(epi_rows),
            pd.DataFrame(morph_rows))


# --------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------


def fmt_ci(v: float, lo: float, hi: float, d: int = 2) -> str:
    if not np.isfinite(v):
        return "–"
    return f"{v:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def markdown(config: DictConfig, sets: pd.DataFrame, concepts: pd.DataFrame,
             epi: pd.DataFrame, morph: pd.DataFrame) -> str:
    lines = [
        "# Disagreement profiles (PANDA Radboud test, all nuclei)",
        "",
        f"Threshold {config.threshold} on seed-mean probabilities (3 seeds). X > Y = X correct, "
        f"Y wrong. Strong = |p_X − p_Y| ≥ {config.strong_disagreement}. 95 % CIs: stratified "
        "slide bootstrap (B = 2,000, seed 20261108). Enrichment = share in the set / share in "
        "the class. Morphometry: (median_set − median_class) / (IQR_class / 1.349).",
        "",
    ]
    for pair in config.pairs:
        lines += [f"## {pair.x_short} vs {pair.y_short} ({pair.question})", ""]
        s = sets[sets["pair"] == pair.name]
        lines += ["### Set sizes", "",
                  "| set | class | subset | n | share of class [95 % CI] | slides |",
                  "|---|---|---|---:|---|---:|"]
        for r in s.itertuples():
            lines.append(f"| {r.direction} | {r.true_class} | {r.subset} | {r.n:,} | "
                         f"{fmt_ci(100 * r.share_of_class, 100 * r.share_ci_lo, 100 * r.share_ci_hi)} % | {r.n_slides} |")
        lines.append("")
        for subset in SUBSETS:
            c = concepts[(concepts["pair"] == pair.name) & (concepts["subset"] == subset)]
            lines += [f"### Dominant-concept enrichment ({subset})", "",
                      "Cells: share in set % / share in class % / enrichment [95 % CI].", ""]
            cols = c[["direction", "true_class"]].drop_duplicates().itertuples(index=False)
            cols = list(cols)
            lines.append("| concept | " + " | ".join(f"{d}, {k}" for d, k in cols) + " |")
            lines.append("|---|" + "---|" * len(cols))
            for g in c["group"].unique():
                cells = []
                for d, k in cols:
                    r = c[(c["group"] == g) & (c["direction"] == d) & (c["true_class"] == k)].iloc[0]
                    cells.append(f"{100 * r.share_set:.1f} / {100 * r.share_class:.1f} / "
                                 f"{fmt_ci(r.enrichment, r.ci_lo, r.ci_hi)}")
                lines.append(f"| {g} | " + " | ".join(cells) + " |")
            lines.append("")
        e = epi[epi["pair"] == pair.name]
        lines += ["### Epithelial share", "",
                  "| set | class | subset | n | set % [95 % CI] | class % [95 % CI] | difference pp [95 % CI] |",
                  "|---|---|---|---:|---|---|---|"]
        for r in e.itertuples():
            lines.append(
                f"| {r.direction} | {r.true_class} | {r.subset} | {r.n:,} | "
                f"{fmt_ci(100 * r.epithelial_share_set, 100 * r.set_ci_lo, 100 * r.set_ci_hi, 1)} | "
                f"{fmt_ci(100 * r.epithelial_share_class, 100 * r.class_ci_lo, 100 * r.class_ci_hi, 1)} | "
                f"{fmt_ci(100 * r.difference, 100 * r.diff_ci_lo, 100 * r.diff_ci_hi, 1)} |")
        lines.append("")
        m = morph[(morph["pair"] == pair.name) & morph["top"]]
        lines += ["### Morphometry: top 5 standardized median differences", "",
                  "| set | class | subset | top 5 (standardized difference) |",
                  "|---|---|---|---|"]
        for (d, k, sub), part in m.groupby(["direction", "true_class", "subset"], sort=False):
            part = part.sort_values("rank")
            cells = ", ".join(f"{r.label} {r.std_difference:+.2f}" for r in part.itertuples())
            lines.append(f"| {d} | {k} | {sub} | {cells} |")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def compute(config: DictConfig) -> tuple[Path, dict]:
    out = Path(config.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    data = load_inputs(config)
    sets, concepts, epi, morph = profile(config, data)
    sets.to_csv(out / "sets.csv", index=False)
    concepts.to_csv(out / "concepts.csv", index=False)
    epi.to_csv(out / "epithelium.csv", index=False)
    morph.to_csv(out / "morphometry.csv", index=False)
    (out / "tables.md").write_text(markdown(config, sets, concepts, epi, morph))

    checks = {
        "nuclei": len(data["y"]),
        "slides": len(data["slide_ids"]),
        "carcinoma_nuclei": int(data["y"].sum()),
        "carcinoma_slide_nuclei_without_label_row": data["labels_missing"],
        "groups": data["group_names"],
        "group_counts_identical_to": config.concepts.run,
        "epithelial_nuclei": int(data["epithelial"].sum()),
        "epithelial_count_identical_to": config.epithelium_run,
        "morphometry_rows_matched": "XAI index rows matched to the predictions by (slide, "
        "nucleus key); labels identical on every nucleus",
        "morphometry_measured": {f: int((~np.isnan(v)).sum()) for f, v in data["features"].items()},
        "threshold": float(config.threshold),
        "strong_disagreement": float(config.strong_disagreement),
        "configurations": configurations(config),
        "resamples": str(config.resamples),
        "accuracy_at_threshold": {
            n: float((((p >= config.threshold) == (data["y"] == 1))).mean())
            for n, p in data["p_mean"].items()
        },
    }
    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=str))

    figure_dirs = [out / "figures"]
    if config.get("repo_figures_dir"):
        figure_dirs.append(Path(config.repo_figures_dir))
    save_disagreement_figures(out, figure_dirs)
    return out, checks


@autolog
def logged(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)
    client = logger.experiment
    client.set_tag(logger.run_id, "code_commit", code_commit())
    for key, value in config.links.items():
        client.set_tag(logger.run_id, key, value)
    out, _ = compute(config)

    sets = pd.read_csv(out / "sets.csv")
    metrics = {}
    for r in sets.itertuples():
        stem = f"{r.pair}/{'fwd' if r.forward else 'rev'}/{r.true_class}/{r.subset}"
        metrics[f"{stem}/n"] = float(r.n)
        metrics[f"{stem}/share_of_class"] = float(r.share_of_class)
    logger.log_metrics(metrics)
    for name in ("sets.csv", "concepts.csv", "epithelium.csv", "morphometry.csv",
                 "tables.md", "checks.json"):
        logger.log_artifact(str(out / name))
    for stem in FIGURES:
        for ext in ("pdf", "png"):
            logger.log_artifact(str(out / "figures" / f"{stem}.{ext}"), "figures")
    for source in SOURCES:
        logger.log_artifact(str(REPO / source), "code")
    (out / "config-resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    logger.log_artifact(str(out / "config-resolved.yaml"), "code")


@with_cli_args(["+postprocessing=disagreement_analysis"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
def main(config: DictConfig) -> None:
    if config.compute_only:
        compute(config)
    else:
        logged(config)


if __name__ == "__main__":
    main()
