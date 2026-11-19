# ruff: noqa: RUF001, RUF002  (mathematical symbols in text are intended)
r"""Q2 "How much do shape, texture, and position contribute?" on epithelial nuclei only.

PANDA Radboud test set, nucleus level. The five configurations of tab:res-sources, three seeds
each, as `postprocessing.results_sources` (the "sources" family of runs a925ae67 / 4081f97a):
position only (blank token, k-NN + RoVE), shape and texture each self-only and with k-NN + RoVE.
Inference outputs only: the stored test predictions via the aligned per-run cache of
`postprocessing.bootstrap_sources`. No training, no inference.

Epithelial nuclei are those of the Q3 context run 027eeaee (`postprocessing.results_context`,
whose `epithelium_classes` is reused): carcinoma (the label) ∪ benign epithelium by the vertex
rule (class 2 of `xai.epithelium_separation`, run 0b24e5db). Their count and prevalence must equal
027eeaee's.

Statistics as there: the Q2 slide resamples (seed 20261108, 2,000, stratified by positive slide;
read, not redrawn) and the 4081f97a logit grid; point = mean of the seeds' exact metrics;
replicates average the seeds within each resample; paired differences are weighted sums within a
resample, percentile CIs, p = 2 x the smaller tail fraction; Holm over the eight AUPRC comparisons
of a subset (all nuclei and epithelial nuclei are separate families). Supporting, outside Holm: the
epithelial minus the all-nuclei value of a comparison (`subset_contrasts`), paired per resample.

Checks (`checks.json`): all nuclei reproduce 4081f97a's replicates, configurations and sources
comparisons; the shared runs' epithelial replicates (shape self-only, thesis model) reproduce
027eeaee's; per-run checks (checkpoint, schedule) are copied from 4081f97a; with an MLflow client,
the training and evaluation commits are checked for the block-mask fix.

Outputs in `out_dir` (no slide ids): `configurations.csv`, `comparisons.csv`,
`subset_contrasts.csv`, `counts.csv`, `runs.csv`, `checks.json`, `replicates.npz`,
`tab_res_sources_epithelium.tex`.
"""

import json
import subprocess
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from tqdm import tqdm

from nuclei_graph.mlflow_utils import tag_parent_run
from postprocessing.bootstrap_sources import (
    METRICS,
    bin_logits,
    bootstrap_p,
    fmt_p,
    holm,
    label_predictions,
    load_labels,
    load_run,
    replicate_metrics,
    slide_histograms,
    to_logits,
)
from postprocessing.results_context import (
    REPO,
    block_mask_rows,
    epithelium_classes,
    load_slides,
    validated_bins,
)
from postprocessing.results_sources import plain, signed


SUBSETS = ("all", "epithelium")
PRIMARY_METRIC = "AUPRC"
SOURCES = (
    "postprocessing/results_sources_epithelium.py",
    "postprocessing/results_context.py",
    "postprocessing/results_sources.py",
    "postprocessing/bootstrap_sources.py",
    "configs/postprocessing/results_sources_epithelium.yaml",
)
PER_RUN_CHECKS = (
    "training_run",
    "eval_date",
    "eval_commit",
    "training_date",
    "checkpoint_epoch",
    "best_epoch",
    "monitor",
    "checkpoint_is_best",
    "checkpoint_source",
    "last_logged_epoch",
    "training_unfinished",
    "schedule_check",
)
# (configuration, row label) of the table, in the order of the ranking question
TABLE_CONFIGS = (
    ("position_knn", "Position only", "k-NN + RoVE"),
    ("shape_self", "Shape", "self-only"),
    ("shape_knn", "Shape", "k-NN + RoVE"),
    ("texture_self", "Texture", "self-only"),
    ("texture_knn", "Texture", "k-NN + RoVE"),
)


# --------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------


def ci_cell(point: float, lo: float, hi: float, sign: bool = False) -> str:
    return f"{signed(point) if sign else plain(point)} [{plain(lo)}, {plain(hi)}]"


def p_cell(p: float, n_resamples: int) -> str:
    return f"$<{2 / n_resamples:.3f}$" if p == 0 else f"{p:.3f}"


def latex_block(
    conf: pd.DataFrame,
    comp: pd.DataFrame,
    contrasts: pd.DataFrame,
    counts: pd.DataFrame,
    config: DictConfig,
    n_resamples: int,
) -> str:
    c = conf[conf["subset"] == "epithelium"].set_index(["configuration", "metric"])
    d = comp[comp["subset"] == "epithelium"].set_index(["comparison", "metric"])
    d_all = comp[comp["subset"] == "all"].set_index(["comparison", "metric"])
    n = counts.set_index("subset").loc["epithelium"]
    ci = f"{config.bootstrap.ci_level:.0%}".replace("%", r"\%")

    conf_rows, notes = [], []
    for name, source, ctx in TABLE_CONFIGS:
        cells = [
            ci_cell(
                c.loc[(name, m), "point"],
                c.loc[(name, m), "ci_lo"],
                c.loc[(name, m), "ci_hi"],
            )
            for m in METRICS
        ]
        conf_rows.append(f"    {source} & {ctx} & " + " & ".join(cells) + r" \\")
        for m in METRICS:
            r = c.loc[(name, m)]
            seeds = ", ".join(f"{r[f'seed_{k}']:.4f}" for k in (1, 2, 3))
            notes.append(f"%   {name}: {m} {r['point']:.4f}, seeds {seeds}")

    comp_rows = []
    for x in config.comparisons:
        a, b = d.loc[(x.name, "AUPRC")], d.loc[(x.name, "AUROC")]
        mark = r"$^{*}$" if a["significant"] == "yes" else ""
        all_a = d_all.loc[(x.name, "AUPRC")]
        comp_rows.append(
            f"    {x.label} & {ci_cell(a['difference'], a['ci_lo'], a['ci_hi'], True)}{mark}"
            f" & {p_cell(a['p_holm'], n_resamples)}"
            f" & {ci_cell(b['difference'], b['ci_lo'], b['ci_hi'], True)}"
            f" & {signed(all_a['difference'])} " + r"\\"
        )
    contrast_notes = [
        f"%   {r.comparison} ({r.metric}): epithelium {r.epithelium:+.4f}, all {r.all:+.4f}, "
        f"epithelium − all {r.difference:+.4f} [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}], "
        f"p {fmt_p(r.p, n_resamples)} (not in the Holm family)"
        for r in contrasts.itertuples()
    ]
    family = "; ".join(x.label for x in config.comparisons)
    return "\n".join(
        [
            "% Generated by postprocessing/results_sources_epithelium.py",
            "% (PANDA Radboud test set, epithelial nuclei as tab:res-context / run 027eeaee).",
            (
                f"% Epithelial test nuclei: {int(n['nuclei']):,}, carcinoma"
                f" {int(n['carcinoma_nuclei']):,}, prevalence {n['prevalence']:.4f}."
            ),
            f"% Holm family (AUPRC): {family}.",
            "% Seed-averaged estimates and the three seeds:",
            *notes,
            "% Shape − texture gap, epithelial vs all nuclei (paired per resample):",
            *contrast_notes,
            r"\begin{table}[t]",
            r"  \centering",
            r"  \caption{Information sources on epithelial nuclei only (carcinoma and benign"
            r" epithelium; "
            + f"{n['nuclei'] / 1e6:.1f}"
            + r"\,M PANDA Radboud test nuclei, prevalence "
            + f"{n['prevalence']:.3f}"
            + r"). Seed-averaged over three seeds with "
            + ci
            + r" paired slide-level bootstrap CIs. (a) Per configuration. (b) Paired differences;"
            r" $p_\mathrm{Holm}$ is Holm-adjusted over the eight AUPRC differences, $^{*}$ $p_\mathrm{Holm} < "
            + f"{config.bootstrap.alpha:g}"
            + r"$; the last column is the AUPRC difference on all test nuclei"
            r" (Table~\ref{tab:res-sources}).}",
            r"  \label{tab:res-sources-epithelium}",
            r"  \small",
            r"  \textbf{(a)}\par\smallskip",
            r"  \begin{tabular}{llcc}",
            r"    \toprule",
            r"    Source & Context & AUPRC [CI] & AUROC [CI] \\",
            r"    \midrule",
            *conf_rows,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"  \par\medskip",
            r"  \textbf{(b)}\par\smallskip",
            r"  \begin{tabular}{lcccc}",
            r"    \toprule",
            (
                r"    Difference & $\Delta$AUPRC [CI] & $p_\mathrm{Holm}$ & $\Delta$AUROC [CI]"
                r" & $\Delta$AUPRC, all \\"
            ),
            r"    \midrule",
            *comp_rows,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )


# --------------------------------------------------------------------------
# Main computation
# --------------------------------------------------------------------------


def compute(config: DictConfig, client: MlflowClient | None) -> tuple[Path, dict]:
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref_dir, ctx_dir = Path(config.reference_dir), Path(config.context_dir)
    b = config.bootstrap

    slides = load_slides(config.metadata_uri)
    slide_ids = slides["slide_id"].tolist()
    labels = load_labels(slides, Path(config.supervision_dir), config.label_column)
    carcinoma = slides["is_carcinoma"].to_numpy(bool)
    names = list(config.configurations)
    runs = [
        (n, i, r)
        for n in names
        for i, r in enumerate(config.configurations[n].eval_runs)
    ]
    assert all(len(config.configurations[n].eval_runs) == 3 for n in names)

    # 1. predictions, labels ----------------------------------------------------
    reference = None
    probs_list = []
    for name, seed_i, run_id in runs:
        df, _ = load_run(
            run_id, slide_ids, Path(config.cache_dir), config.predictions_artifact
        )
        y, _ = label_predictions(df, labels, carcinoma)
        if reference is None:
            reference, y_ref, slide = df[["slide", "key"]], y, df["slide"].to_numpy()
        assert np.array_equal(df["slide"].to_numpy(), reference["slide"].to_numpy())
        assert np.array_equal(df["key"].to_numpy(), reference["key"].to_numpy()), (
            f"{run_id}: other nuclei than the first run"
        )
        assert np.array_equal(y, y_ref)
        probs_list.append(df["p"].to_numpy())
        print(f"{name} seed {seed_i + 1} {run_id[:8]}: loaded", flush=True)
    probs = np.stack(probs_list)
    del probs_list, reference
    y = y_ref
    n_slides = len(slides)
    positive = np.bincount(slide, weights=(y == 1), minlength=n_slides) > 0

    # 2. epithelial nuclei (as 027eeaee) ----------------------------------------
    cls, label_vs_class = epithelium_classes(config, slides, slide, y)
    epi = (y == 1) | (cls == 2)
    del cls
    xai_counts = pd.read_csv(config.epithelium.counts).set_index("split").loc["test"]
    assert int(epi.sum()) == int(xai_counts["n_epithelial"])
    assert int((y == 1).sum()) == int(xai_counts["n_carcinoma"])
    masks = {"all": np.ones(len(y), bool), "epithelium": epi}

    # 3. resamples ----------------------------------------------------------------
    saved = np.load(config.resamples)
    assert saved["slide_ids"].tolist() == slide_ids, "resamples: different slides"
    assert np.array_equal(saved["positive"], positive), "resamples: different strata"
    weights = saved["multiplicities"].astype(np.float64)
    n_resamples = len(weights)

    # 4. per subset: histograms, replicates -----------------------------------------
    ref_checks = json.loads((ref_dir / "checks.json").read_text())
    lo, hi = ref_checks["histogram"]["logit_range_q2"]
    rep = {s: {m: np.empty((len(runs), n_resamples)) for m in METRICS} for s in SUBSETS}
    exact: dict[str, list[dict[str, float]]] = {}
    counts, hist_checks = [], {}
    for subset in SUBSETS:
        mask = masks[subset]
        ys, ss = y[mask], slide[mask]
        n_bins, exact[subset], max_err, _ = validated_bins(
            probs[:, mask], ys, lo, hi, int(ref_checks["histogram"]["n_bins"]), config
        )
        hist_checks[subset] = {"n_bins": n_bins, "max_abs_error": max_err}
        for i in tqdm(range(len(runs)), desc=f"replicates ({subset})"):
            bins = bin_logits(to_logits(probs[i, mask]), lo, hi, n_bins)
            pos, neg = slide_histograms(ss, bins, ys, n_slides, n_bins)
            for m, v in replicate_metrics(pos, neg, weights).items():
                rep[subset][m][i] = v
        slide_n = np.bincount(ss, minlength=n_slides)
        counts.append(
            {
                "subset": subset,
                "slides_with_nuclei": int((slide_n > 0).sum()),
                "slides_with_carcinoma": int(positive.sum()),
                "nuclei": int(mask.sum()),
                "carcinoma_nuclei": int((ys == 1).sum()),
                "non_carcinoma_nuclei": int((ys == 0).sum()),
                "prevalence": float(np.mean(ys == 1)),
            }
        )
    np.savez_compressed(
        out / "replicates.npz",
        eval_runs=np.array([r for _, _, r in runs]),
        configurations=np.array([n for n, _, _ in runs]),
        **{f"{s}/{m}": rep[s][m] for s in SUBSETS for m in METRICS},  # type: ignore[arg-type]
    )
    counts_df = pd.DataFrame(counts)
    counts_df.to_csv(out / "counts.csv", index=False)
    ctx_counts = pd.read_csv(ctx_dir / "counts.csv").set_index("subset")
    counts_match = {
        s: {
            k: bool(counts_df.set_index("subset").loc[s, k] == ctx_counts.loc[s, k])
            for k in ("nuclei", "carcinoma_nuclei", "slides_with_nuclei", "prevalence")
        }
        for s in SUBSETS
    }
    assert all(all(v.values()) for v in counts_match.values()), counts_match

    # 5. configurations -----------------------------------------------------------
    pct = (1 - b.ci_level) / 2 * 100
    avg_rep: dict = {s: {m: {} for m in METRICS} for s in SUBSETS}
    avg_point: dict = {s: {m: {} for m in METRICS} for s in SUBSETS}
    conf_rows = []
    for subset in SUBSETS:
        for name in names:
            idx = [i for i, (n, _, _) in enumerate(runs) if n == name]
            for m in METRICS:
                per_seed = np.array([exact[subset][i][m] for i in idx])
                avg_rep[subset][m][name] = rep[subset][m][idx].mean(0)
                avg_point[subset][m][name] = float(per_seed.mean())
                ci_lo, ci_hi = np.percentile(avg_rep[subset][m][name], [pct, 100 - pct])
                conf_rows.append(
                    {
                        "subset": subset,
                        "configuration": name,
                        "label": config.configurations[name].label,
                        "metric": m,
                        "n_seeds": len(idx),
                        "point": avg_point[subset][m][name],
                        "ci_lo": float(ci_lo),
                        "ci_hi": float(ci_hi),
                        "seed_sd": float(per_seed.std(ddof=1)),
                        **{f"seed_{k + 1}": float(v) for k, v in enumerate(per_seed)},
                        **{f"run_{k + 1}": runs[i][2] for k, i in enumerate(idx)},
                    }
                )
    conf = pd.DataFrame(conf_rows)
    conf.to_csv(out / "configurations.csv", index=False)

    # 6. paired comparisons, Holm per subset over the AUPRC comparisons -----------
    diffs: dict = {s: {m: {} for m in METRICS} for s in SUBSETS}
    comp_rows = []
    for subset in SUBSETS:
        for m in METRICS:
            rows = []
            for x in config.comparisons:
                diff = sum(w * avg_rep[subset][m][n] for n, w in x.weights.items())
                point = sum(w * avg_point[subset][m][n] for n, w in x.weights.items())
                diffs[subset][m][x.name] = (float(point), diff)
                ci_lo, ci_hi = np.percentile(diff, [pct, 100 - pct])
                p = bootstrap_p(diff)
                rows.append(
                    {
                        "subset": subset,
                        "family": f"sources_{subset}",
                        "status": "final" if m == PRIMARY_METRIC else "supporting",
                        "metric": m,
                        "comparison": x.name,
                        "label": x.label,
                        "difference": float(point),
                        "ci_lo": float(ci_lo),
                        "ci_hi": float(ci_hi),
                        "ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
                        "replicate_mean": float(diff.mean()),
                        "p": p,
                        "p_display": fmt_p(p, n_resamples),
                    }
                )
            if m == PRIMARY_METRIC:
                adjusted = holm(np.array([r["p"] for r in rows]))
                for r, a in zip(rows, adjusted, strict=True):
                    r["p_holm"] = float(a)
                    r["p_holm_display"] = fmt_p(float(a), n_resamples)
                    r["significant"] = "yes" if a < b.alpha else "no"
            comp_rows += rows
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(out / "comparisons.csv", index=False)

    # 7. epithelial minus all nuclei (supporting, outside Holm) ---------------------
    contrast_rows = []
    for name in config.subset_contrasts:
        for m in METRICS:
            (pe, de), (pa, da) = diffs["epithelium"][m][name], diffs["all"][m][name]
            delta = de - da
            ci_lo, ci_hi = np.percentile(delta, [pct, 100 - pct])
            p = bootstrap_p(delta)
            contrast_rows.append(
                {
                    "comparison": name,
                    "metric": m,
                    "epithelium": pe,
                    "all": pa,
                    "difference": pe - pa,
                    "ci_lo": float(ci_lo),
                    "ci_hi": float(ci_hi),
                    "p": p,
                    "p_display": fmt_p(p, n_resamples),
                    "abs_gap_epithelium_minus_all": abs(pe) - abs(pa),
                }
            )
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(out / "subset_contrasts.csv", index=False)

    # 8. against run 4081f97a (all nuclei) -------------------------------------------
    with np.load(ref_dir / "replicates.npz") as z:
        ref_runs = z["eval_runs"].tolist()
        rep_diff_all = {
            m: max(
                float(np.abs(rep["all"][m][i] - z[m][ref_runs.index(r)]).max())
                for i, (_, _, r) in enumerate(runs)
            )
            for m in METRICS
        }
    ref_conf = pd.read_csv(ref_dir / "configurations.csv")
    merged = conf[conf["subset"] == "all"].merge(
        ref_conf, on=["configuration", "metric"], suffixes=("", "_ref")
    )
    assert len(merged) == 2 * len(names)
    assert (
        merged[[f"run_{k}" for k in (1, 2, 3)]].to_numpy()
        == merged[[f"run_{k}_ref" for k in (1, 2, 3)]].to_numpy()
    ).all(), "4081f97a: other evaluation runs"
    cols = ["point", "ci_lo", "ci_hi", "seed_1", "seed_2", "seed_3"]
    ref_comp = pd.read_csv(ref_dir / "comparisons.csv")
    ref_comp = ref_comp[ref_comp["family"] == config.reference_family]
    merged_comp = comp[comp["subset"] == "all"].merge(
        ref_comp, on=["comparison", "metric"], suffixes=("", "_ref")
    )
    assert len(merged_comp) == 2 * len(config.comparisons)
    for x in config.comparisons:  # same weights as the reference family
        assert x.name in set(ref_comp["comparison"])
    comp_cols = ["difference", "ci_lo", "ci_hi", "p"]
    holm_rows = merged_comp[merged_comp["metric"] == PRIMARY_METRIC]

    # 9. against run 027eeaee (epithelial replicates of the shared runs) -----------
    with np.load(ctx_dir / "replicates.npz") as z:
        ctx_runs = z["eval_runs"].tolist()
        shared = [
            (i, ctx_runs.index(r)) for i, (_, _, r) in enumerate(runs) if r in ctx_runs
        ]
        rep_diff_epi = {
            m: max(
                float(np.abs(rep["epithelium"][m][i] - z[f"epithelium/{m}"][j]).max())
                for i, j in shared
            )
            for m in METRICS
        }
    ctx_conf = pd.read_csv(ctx_dir / "configurations.csv")
    ctx_conf = ctx_conf[ctx_conf["subset"] == "epithelium"]
    ctx_conf["configuration"] = ctx_conf["configuration"].map(
        {v: k for k, v in config.context_names.items()}
    )
    merged_ctx = conf[conf["subset"] == "epithelium"].merge(
        ctx_conf.dropna(subset=["configuration"]),
        on=["configuration", "metric"],
        suffixes=("", "_ref"),
    )
    assert len(merged_ctx) == 2 * len(config.context_names)

    # 10. per-run checks of run 4081f97a, training runs as in the config ------------
    ref_run_checks = pd.read_csv(ref_dir / "checks.csv").set_index("eval_run")
    run_rows = []
    for i, (name, seed_i, run_id) in enumerate(runs):
        r = ref_run_checks.loc[run_id]
        expected = config.configurations[name].training_runs[seed_i]
        assert str(r["training_run"]).startswith(expected), (run_id, expected)
        run_rows.append(
            {
                "configuration": name,
                "seed": seed_i + 1,
                "eval_run": run_id,
                **{k: r[k] for k in PER_RUN_CHECKS},
                **{f"{m}_{s}": exact[s][i][m] for s in SUBSETS for m in METRICS},
            }
        )
    runs_df = pd.DataFrame(run_rows)
    block_mask: str | dict = "not checked (no MLflow client)"
    if client is not None:
        bm = pd.DataFrame(
            block_mask_rows(client, runs_df, config.block_mask_fix_commit)
        )
        runs_df = runs_df.merge(bm, on="eval_run")
        block_mask = {
            name: runs_df.loc[
                runs_df["configuration"] == name,
                [
                    "seed",
                    "training_commit",
                    "training_has_fix",
                    "eval_commit",
                    "eval_has_fix",
                ],
            ].to_dict("records")
            for name in names
        }
    runs_df.to_csv(out / "runs.csv", index=False)

    # 11. table, ranking, checks -----------------------------------------------------
    (out / "tab_res_sources_epithelium.tex").write_text(
        latex_block(conf, comp, contrasts, counts_df, config, n_resamples)
    )
    epi_comp = comp[
        (comp["subset"] == "epithelium") & (comp["metric"] == PRIMARY_METRIC)
    ]
    epi_comp = epi_comp.set_index("comparison")

    def holds(name: str, sign: int) -> bool:  # Holm-significant with the given sign
        r = epi_comp.loc[name]
        return bool(r["significant"] == "yes" and sign * r["difference"] > 0)

    ranking = {
        "texture_gt_shape_with_context": holds("shape_vs_texture_knn", -1),
        "shape_gt_position_with_context": holds("shape_vs_position_knn", 1),
        "texture_gt_position_with_context": holds("texture_vs_position_knn", 1),
    }
    ranking["texture_gt_shape_gt_position"] = all(ranking.values())
    ranking["texture_gt_shape_self_only"] = holds("shape_vs_texture_self", -1)

    checks = {
        "counts": counts_df.set_index("subset").to_dict("index"),
        "counts_equal_to_context_run": counts_match,
        "labels": "annot_label left-joined on (slide_id, id), missing -> 0, "
        "is_carcinoma = False -> 0 (as the evaluation and runs 4081f97a, 027eeaee)",
        "epithelium": {
            "rule": "carcinoma = label 1; benign epithelium = vertex-rule class 2 "
            "(>= 50 % of polygon vertices in mask value 2); as run 027eeaee",
            "classes_dir": str(config.epithelium.classes_dir),
            **label_vs_class,
            "n_epithelial_equals_xai_counts": True,
        },
        "resamples": {
            "file": str(config.resamples),
            "n_resamples": n_resamples,
            "seed": int(saved["seed"]),
            "slides_identical": True,
            "strata_identical": True,
        },
        "histogram": {
            "logit_range": [lo, hi],
            "tolerance": float(config.histogram.tolerance),
            **hist_checks,
        },
        "reference_run_all_nuclei": {
            "run": config.reference_run,
            "family": config.reference_family,
            "max_abs_replicate_difference": rep_diff_all,
            "max_abs_configuration_difference": float(
                np.abs(
                    merged[cols].to_numpy(float)
                    - merged[[f"{c}_ref" for c in cols]].to_numpy(float)
                ).max()
            ),
            "max_abs_comparison_difference": float(
                np.abs(
                    merged_comp[comp_cols].to_numpy(float)
                    - merged_comp[[f"{c}_ref" for c in comp_cols]].to_numpy(float)
                ).max()
            ),
            "max_abs_p_holm_difference": float(
                np.abs(holm_rows["p_holm"] - holm_rows["p_holm_ref"]).max()
            ),
            "significance_identical": bool(
                (holm_rows["significant"] == holm_rows["significant_ref"]).all()
            ),
        },
        "context_run_epithelium": {
            "run": config.context_run,
            "shared_runs": [runs[i][2] for i, _ in shared],
            "max_abs_replicate_difference": rep_diff_epi,
            "max_abs_configuration_difference": float(
                np.abs(
                    merged_ctx[cols].to_numpy(float)
                    - merged_ctx[[f"{c}_ref" for c in cols]].to_numpy(float)
                ).max()
            ),
        },
        "per_run_checks_from_reference_run": {
            "checkpoint_not_best": runs_df.loc[
                ~runs_df["checkpoint_is_best"], "eval_run"
            ].tolist(),
            "training_unfinished": runs_df.loc[
                runs_df["training_unfinished"],
                ["eval_run", "checkpoint_epoch", "last_logged_epoch"],
            ].to_dict("records"),
            "schedule_not_ok": runs_df.loc[
                runs_df["schedule_check"] != "ok", "eval_run"
            ].tolist(),
        },
        "block_mask_fix": {"commit": config.block_mask_fix_commit, "runs": block_mask},
        "holm_family": {
            s: [f"{x.label} (AUPRC)" for x in config.comparisons] for s in SUBSETS
        },
        "ranking_epithelium_auprc": ranking,
    }
    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=str))
    return out, checks


def code_commit() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no", "--", *SOURCES],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{commit}{' (dirty)' if dirty else ''}"


@with_cli_args(["+postprocessing=results_sources_epithelium"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)
    client = logger.experiment
    client.set_tag(logger.run_id, "code_commit", code_commit())
    client.set_tag(logger.run_id, "reference_run", config.reference_run)
    client.set_tag(logger.run_id, "context_run", config.context_run)
    for name, c in config.configurations.items():
        client.set_tag(logger.run_id, f"eval_runs.{name}", ",".join(c.eval_runs))
    out, checks = compute(config, client)

    conf = pd.read_csv(out / "configurations.csv")
    comp = pd.read_csv(out / "comparisons.csv")
    contrasts = pd.read_csv(out / "subset_contrasts.csv")
    metrics = {
        f"counts/{s}/{k}": float(v)
        for s, row in checks["counts"].items()
        for k, v in row.items()
    }
    for r in conf.itertuples():
        for key in ("point", "ci_lo", "ci_hi", "seed_sd"):
            metrics[f"test/{r.subset}/{r.configuration}/{r.metric}_{key}"] = getattr(
                r, key
            )
    for r in comp.itertuples():
        for key in ("difference", "ci_lo", "ci_hi", "p", "p_holm"):
            value = getattr(r, key, None)
            if value is not None and pd.notna(value):
                metrics[f"diff/{r.subset}/{r.comparison}/{r.metric}_{key}"] = value
    for r in contrasts.itertuples():
        for key in ("difference", "ci_lo", "ci_hi", "p"):
            metrics[f"epithelium_minus_all/{r.comparison}/{r.metric}_{key}"] = getattr(
                r, key
            )
    ref = checks["reference_run_all_nuclei"]
    metrics["check/reference_max_abs_replicate_difference"] = max(
        ref["max_abs_replicate_difference"].values()
    )
    metrics["check/reference_max_abs_comparison_difference"] = ref[
        "max_abs_comparison_difference"
    ]
    metrics["check/context_max_abs_replicate_difference"] = max(
        checks["context_run_epithelium"]["max_abs_replicate_difference"].values()
    )
    logger.log_metrics(metrics)

    for name in (
        *sorted(p.name for p in out.glob("*.csv")),
        "checks.json",
        "tab_res_sources_epithelium.tex",
        "replicates.npz",
    ):
        logger.log_artifact(str(out / name))
    for source in SOURCES:
        logger.log_artifact(str(REPO / source), "code")
    (out / "config-resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    logger.log_artifact(str(out / "config-resolved.yaml"), "code")


if __name__ == "__main__":
    main()
