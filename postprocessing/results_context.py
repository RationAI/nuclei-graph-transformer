# ruff: noqa: RUF002  (mathematical symbols in text are intended)
r"""Q3 "How important is tissue context?": tab:res-context, fig:res-context, epithelial nuclei.

PANDA Radboud test set, nucleus level; shape embedding without position encoding, three seeds
each: self-only, k-NN attention (k = 24) and dense attention, plus the thesis model (k-NN + RoVE)
as a reference line. Inference outputs only: the test predictions stored in existing evaluation
runs (experiment 37), via the aligned per-run cache of `postprocessing.bootstrap_sources`. No
training, no inference.

Statistics as `postprocessing.results_sources` (run 4081f97a, the Q2/Q3 positions run), whose
helpers, slide resamples (seed 20261108, 2,000, stratified by positive slide; read, not
redrawn) and logit grid are reused: the point estimate is the mean of the seeds' exact metrics;
replicates average the seeds within each resample (histogram metrics, nuclei weighted by their
slide's multiplicity); paired differences are weighted sums within a resample with percentile
CIs and p = 2 x the smaller tail fraction; Holm over the AUPRC comparisons of a family.

Two subsets, the same resamples:
- all: every test nucleus (labels as the evaluation: annot_label, missing -> 0). Family = run
  4081f97a's "context" family: k-NN − self-only, dense − k-NN, dense − self-only (AUPRC). The
  replicates, configurations and comparisons are compared with 4081f97a's (checks.json).
- epithelium: carcinoma ∪ benign epithelium by the vertex rule, as `xai.epithelium_separation`
  (run 0b24e5db): carcinoma = the nucleus label, benign epithelium = ≥ 50 % of the polygon
  vertices in mask value 2. Its per-slide class cache (id order) is read; the label must equal
  "class ≥ 3" on every nucleus. A separate family with the same three comparisons. A resample
  keeps the slide multiplicities of the all-nuclei resample (strata = slides with carcinoma
  nuclei), so slides without epithelial nuclei contribute nothing.

Per-run checks (checkpoint, schedule, logged metrics) are those of run 4081f97a, copied into
`runs.csv` with the training runs compared with the config. With an MLflow client, the
training commit of each run is read and both commits are checked for the block-mask fix
(`block_mask_fix_commit`, git ancestry).

Outputs in `out_dir` (no slide ids): `configurations.csv`, `comparisons.csv`, `counts.csv`,
`runs.csv`, `checks.json`, `replicates.npz`, `pr_curves.npz`, `tab_res_context.tex`,
`tab_res_context_epithelium.tex`, `figures/res_context.{pdf,png}`.
"""

import json
import subprocess
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from tqdm import tqdm

from postprocessing.bootstrap_sources import (
    METRICS,
    bin_logits,
    bootstrap_p,
    exact_metrics,
    fmt_p,
    hist_metrics,
    holm,
    label_predictions,
    load_labels,
    load_run,
    replicate_metrics,
    slide_histograms,
    to_logits,
)
from postprocessing.results_sources import COMMIT_TAGS, plain, pr_curve
from visualization.context_figure import save_context_figure


REPO = Path(__file__).parents[1]
SUBSETS = ("all", "epithelium")
PRIMARY_METRIC = "AUPRC"
TABLE_ROWS = (("self", None), ("knn", "knn_vs_self"), ("dense", "dense_vs_knn"))


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def load_slides(uri: str) -> pd.DataFrame:
    slides = pd.read_parquet(download_artifacts(uri))
    if "split" in slides.columns:  # a table of several splits (dry runs without MLflow)
        slides = slides[slides["split"] == "test"]
    return slides[["slide_id", "is_carcinoma"]].sort_values(
        "slide_id", ignore_index=True
    )


def epithelium_classes(
    config: DictConfig, slides: pd.DataFrame, slide: NDArray, y: NDArray
) -> tuple[NDArray[np.int8], dict]:
    """Vertex-rule class of every nucleus, in the (slide, key) order of the predictions.

    The cache stores each slide's classes in nucleus-id order. Keys are the first 64 bits of the
    hex ids, unique within a slide, so key order is id order.
    """
    e = config.epithelium
    xai_slides = pd.read_parquet(e.slides)
    xai_slides = xai_slides[xai_slides["split"] == "test"].set_index("slide_id")
    assert set(xai_slides.index) == set(slides["slide_id"]), "class cache: other slides"
    assert (
        xai_slides.loc[slides["slide_id"], "is_carcinoma"].to_numpy(bool)
        == slides["is_carcinoma"].to_numpy(bool)
    ).all(), "class cache: is_carcinoma differs"
    per_slide = np.bincount(slide, minlength=len(slides))
    assert (
        xai_slides.loc[slides["slide_id"], "n_nuclei"].to_numpy() == per_slide
    ).all(), "class cache: nuclei per slide differ from the predictions"
    cls = np.concatenate(
        [np.load(Path(e.classes_dir) / f"{s}.npy") for s in slides["slide_id"]]
    ).astype(np.int8)
    assert len(cls) == len(y)
    label_vs_class = {
        "label1_but_class_below_3": int(((y == 1) & (cls < 3)).sum()),
        "label0_but_class_3_to_5": int(((y == 0) & (cls >= 3)).sum()),
    }
    assert not any(label_vs_class.values()), label_vs_class
    return cls, label_vs_class


def validated_bins(
    probs: NDArray, y: NDArray, lo: float, hi: float, n_bins: int, config: DictConfig
) -> tuple[int, list[dict], float, list[dict]]:
    """The reference grid, doubled until every run's histogram metrics match the exact ones."""
    exact = [exact_metrics(y, p) for p in probs]
    while True:
        errors = []
        for p, e in zip(probs, exact, strict=True):
            bins = bin_logits(to_logits(p), lo, hi, n_bins)
            full = hist_metrics(
                np.bincount(bins[y == 1], minlength=n_bins).astype(np.float64),
                np.bincount(bins[y == 0], minlength=n_bins).astype(np.float64),
            )
            errors.append({m: float(full[m] - e[m]) for m in METRICS})
        max_err = max(abs(v) for err in errors for v in err.values())
        print(f"histogram: {n_bins} bins, max |error| {max_err:.2e}", flush=True)
        if max_err <= config.histogram.tolerance or n_bins >= config.histogram.max_bins:
            break
        n_bins *= 2
    assert max_err <= config.histogram.tolerance, "histogram grid too coarse"
    return n_bins, exact, max_err, errors


# --------------------------------------------------------------------------
# Block-mask fix
# --------------------------------------------------------------------------


def contains_commit(commit: str | None, fix: str) -> bool | None:
    if not commit:
        return None
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", fix, commit],
        cwd=REPO,
        capture_output=True,
        check=False,
    )
    return {0: True, 1: False}.get(result.returncode)  # other: commit not in this clone


def block_mask_rows(client: MlflowClient, runs: pd.DataFrame, fix: str) -> list[dict]:
    rows = []
    for r in runs.itertuples():
        tags = client.get_run(r.training_run).data.tags
        commit = next((tags[k] for k in COMMIT_TAGS if k in tags), "")
        train_commit = commit.split()[0][:8] if commit else None
        rows.append(
            {
                "eval_run": r.eval_run,
                "training_commit": train_commit,
                "training_has_fix": contains_commit(train_commit, fix),
                "eval_has_fix": contains_commit(str(r.eval_commit), fix),
            }
        )
    return rows


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def table_rows(
    conf: pd.DataFrame, comp: pd.DataFrame, config: DictConfig, subset: str
) -> str:
    c = conf[conf["subset"] == subset].set_index(["configuration", "metric"])
    d = comp[(comp["subset"] == subset) & (comp["metric"] == PRIMARY_METRIC)]
    d = d.set_index("comparison")

    def delta(name: str) -> str:
        r = d.loc[name]
        mark = "^{*}" if r["significant"] == "yes" else ""
        sign = "+" if r["difference"] >= 0 else "-"
        return f"${sign}{abs(r['difference']):.3f}{mark}$ [{plain(r['ci_lo'])}, {plain(r['ci_hi'])}]"

    rows, notes = [], []
    for name, cmp in TABLE_ROWS:
        auprc, auroc = c.loc[(name, "AUPRC")], c.loc[(name, "AUROC")]
        rows.append(
            f"    {config.configurations[name].label} & {auprc['point']:.3f} & "
            f"{auroc['point']:.3f} & {delta(cmp) if cmp else '--'} " + r"\\"
        )
        for m, r in (("AUPRC", auprc), ("AUROC", auroc)):
            seeds = ", ".join(f"{r[f'seed_{k}']:.4f}" for k in (1, 2, 3))
            notes.append(
                f"%   {config.configurations[name].short}: {m} {r['point']:.4f} "
                f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}], seeds {seeds}"
            )
    family = "; ".join(f"{x.label} (AUPRC)" for x in config.comparisons)
    ci = f"{config.bootstrap.ci_level:.0%}"
    what = "all test nuclei" if subset == "all" else "epithelial nuclei only"
    return "\n".join(
        [
            f"% Generated by postprocessing/results_context.py (PANDA Radboud test set, {what}).",
            "% Shape embedding, no position encoding; seed-averaged over three seeds.",
            (
                f"% Delta = AUPRC difference to the row above, {ci} paired slide-level bootstrap CI"
                f" (2,000 resamples, stratified); * = Holm-adjusted p < {config.bootstrap.alpha:g}"
                f" over the family: {family}."
            ),
            "% Seed-averaged estimates with CIs and the three seeds:",
            *notes,
            *rows,
            "",
        ]
    )


# --------------------------------------------------------------------------
# Main computation
# --------------------------------------------------------------------------


def compute(config: DictConfig, client: MlflowClient | None) -> tuple[Path, dict]:
    out = Path(config.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    ref_dir = Path(config.reference_dir)
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
        probs_list.append(df["p"].to_numpy())
        print(f"{name} seed {seed_i + 1} {run_id[:8]}: loaded", flush=True)
    probs = np.stack(probs_list)
    del probs_list
    y = y_ref
    n_slides = len(slides)
    positive = np.bincount(slide, weights=(y == 1), minlength=n_slides) > 0

    # 2. epithelial nuclei ------------------------------------------------------
    cls, label_vs_class = epithelium_classes(config, slides, slide, y)
    epi = (y == 1) | (cls == 2)
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

    # 4. per subset: histograms, replicates, curves -------------------------------
    ref_checks = json.loads((ref_dir / "checks.json").read_text())
    lo, hi = ref_checks["histogram"]["logit_range_q2"]
    grid = np.linspace(0, 1, int(config.figure.recall_grid))
    rep = {s: {m: np.empty((len(runs), n_resamples)) for m in METRICS} for s in SUBSETS}
    exact: dict[str, list[dict[str, float]]] = {}
    curves: dict[str, NDArray] = {}
    counts: list[dict] = []
    hist_checks: dict[str, dict] = {}
    for subset in SUBSETS:
        mask = masks[subset]
        ys, ss = y[mask], slide[mask]
        n_bins, exact[subset], max_err, _ = validated_bins(
            probs[:, mask], ys, lo, hi, int(ref_checks["histogram"]["n_bins"]), config
        )
        hist_checks[subset] = {"n_bins": n_bins, "max_abs_error": max_err}
        per_config: dict[str, list[NDArray]] = {n: [] for n in names}
        for i, p in enumerate(tqdm(probs[:, mask], desc=f"replicates ({subset})")):
            bins = bin_logits(to_logits(p), lo, hi, n_bins)
            pos, neg = slide_histograms(ss, bins, ys, n_slides, n_bins)
            per_config[runs[i][0]].append(pr_curve(pos.sum(0), neg.sum(0), grid))
            for m, v in replicate_metrics(pos, neg, weights).items():
                rep[subset][m][i] = v
        curves |= {f"{subset}/{n}": np.stack(c) for n, c in per_config.items()}
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
    arrays: dict[str, NDArray] = {
        "eval_runs": np.array([r for _, _, r in runs]),
        "configurations": np.array([n for n, _, _ in runs]),
        **{f"{s}/{m}": rep[s][m] for s in SUBSETS for m in METRICS},
    }
    np.savez_compressed(out / "replicates.npz", **arrays)  # type: ignore[arg-type]
    np.savez_compressed(out / "pr_curves.npz", recall=grid, **curves)  # type: ignore[arg-type]
    counts_df = pd.DataFrame(counts)
    counts_df.to_csv(out / "counts.csv", index=False)

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
    comp_rows = []
    for subset in SUBSETS:
        for m in METRICS:
            rows = []
            for c in config.comparisons:
                diff = sum(w * avg_rep[subset][m][n] for n, w in c.weights.items())
                point = sum(w * avg_point[subset][m][n] for n, w in c.weights.items())
                ci_lo, ci_hi = np.percentile(diff, [pct, 100 - pct])
                p = bootstrap_p(diff)
                rows.append(
                    {
                        "subset": subset,
                        "family": f"context_{subset}",
                        "status": "final" if m == PRIMARY_METRIC else "supporting",
                        "metric": m,
                        "comparison": c.name,
                        "label": c.label,
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

    # 7. against run 4081f97a -------------------------------------------------------
    ref_names = dict(config.reference_names)
    with np.load(ref_dir / "replicates.npz") as z:
        ref_runs = z["eval_runs"].tolist()
        rep_diff = {
            m: max(
                float(np.abs(rep["all"][m][i] - z[m][ref_runs.index(r)]).max())
                for i, (_, _, r) in enumerate(runs)
            )
            for m in METRICS
        }
    ref_conf = pd.read_csv(ref_dir / "configurations.csv")
    ref_conf["configuration"] = ref_conf["configuration"].map(
        {v: k for k, v in ref_names.items()}
    )
    merged = conf[conf["subset"] == "all"].merge(
        ref_conf.dropna(subset=["configuration"]),
        on=["configuration", "metric"],
        suffixes=("", "_ref"),
    )
    cols = ["point", "ci_lo", "ci_hi", "seed_1", "seed_2", "seed_3"]
    assert len(merged) == 2 * len(names)
    assert (
        merged[[f"run_{k}" for k in (1, 2, 3)]].to_numpy()
        == merged[[f"run_{k}_ref" for k in (1, 2, 3)]].to_numpy()
    ).all()
    ref_comp = pd.read_csv(ref_dir / "comparisons.csv")
    ref_comp = ref_comp[ref_comp["family"] == "context"]
    ref_comp["comparison"] = ref_comp["comparison"].map(
        {v: k for k, v in config.reference_comparisons.items()}
    )
    merged_comp = comp[comp["subset"] == "all"].merge(
        ref_comp, on=["comparison", "metric"], suffixes=("", "_ref")
    )
    assert len(merged_comp) == 2 * len(config.comparisons)
    comp_cols = ["difference", "ci_lo", "ci_hi", "p"]
    holm_rows = merged_comp[merged_comp["metric"] == PRIMARY_METRIC]

    # 8. per-run checks of run 4081f97a, training runs as in the config ---------------
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
                **{
                    k: r[k]
                    for k in (
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
                },
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

    # 9. tables, figure, checks ---------------------------------------------------------
    (out / "tab_res_context.tex").write_text(table_rows(conf, comp, config, "all"))
    (out / "tab_res_context_epithelium.tex").write_text(
        table_rows(conf, comp, config, "epithelium")
    )
    figure_dirs = [out / "figures"]
    if config.get("repo_figures_dir"):
        figure_dirs.append(Path(config.repo_figures_dir))
    save_context_figure(out, figure_dirs)

    thesis = conf[(conf["subset"] == "all") & (conf["configuration"] == "thesis")]
    thesis = thesis.set_index("metric")["point"]
    checks = {
        "counts": counts_df.set_index("subset").to_dict("index"),
        "labels": "annot_label left-joined on (slide_id, id), missing -> 0, "
        "is_carcinoma = False -> 0 (as the evaluation and run 4081f97a)",
        "epithelium": {
            "rule": "carcinoma = label 1; benign epithelium = vertex-rule class 2 "
            "(>= 50 % of polygon vertices in mask value 2); xai.epithelium_separation (0b24e5db)",
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
        "reference_run": {
            "run": config.reference_run,
            "max_abs_replicate_difference": rep_diff,
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
            "family_identical": set(ref_comp["comparison"])
            == {c.name for c in config.comparisons},
        },
        "thesis_model_vs_tab_res_nucleus": {
            m: {
                "point": float(thesis[m]),
                "rounded": round(float(thesis[m]), 3),
                "table": float(config.thesis_reference[m]),
                "reproduced": round(float(thesis[m]), 3)
                == float(config.thesis_reference[m]),
            }
            for m in METRICS
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
            s: [f"{c.label} (AUPRC)" for c in config.comparisons] for s in SUBSETS
        },
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
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            "postprocessing/results_context.py",
            "postprocessing/results_sources.py",
            "postprocessing/bootstrap_sources.py",
            "visualization/context_figure.py",
            "visualization/thesis_results_figures.py",
            "configs/postprocessing/results_context.yaml",
        ],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{commit}{' (dirty)' if dirty else ''}"


@with_cli_args(["+postprocessing=results_context"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    client = logger.experiment
    client.set_tag(logger.run_id, "code_commit", code_commit())
    client.set_tag(logger.run_id, "reference_run", config.reference_run)
    for name, c in config.configurations.items():
        client.set_tag(logger.run_id, f"eval_runs.{name}", ",".join(c.eval_runs))
    out, checks = compute(config, client)

    conf = pd.read_csv(out / "configurations.csv")
    comp = pd.read_csv(out / "comparisons.csv")
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
    ref = checks["reference_run"]
    metrics["check/reference_max_abs_replicate_difference"] = max(
        ref["max_abs_replicate_difference"].values()
    )
    metrics["check/reference_max_abs_comparison_difference"] = ref[
        "max_abs_comparison_difference"
    ]
    logger.log_metrics(metrics)

    for name in (
        *sorted(p.name for p in out.glob("*.csv")),
        "checks.json",
        "tab_res_context.tex",
        "tab_res_context_epithelium.tex",
        "replicates.npz",
        "pr_curves.npz",
    ):
        logger.log_artifact(str(out / name))
    for name in ("res_context.pdf", "res_context.png"):
        logger.log_artifact(str(out / "figures" / name), "figures")
    for source in (
        "postprocessing/results_context.py",
        "postprocessing/results_sources.py",
        "postprocessing/bootstrap_sources.py",
        "visualization/context_figure.py",
        "visualization/thesis_results_figures.py",
        "configs/postprocessing/results_context.yaml",
    ):
        logger.log_artifact(str(REPO / source), "code")
    (out / "config-resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    logger.log_artifact(str(out / "config-resolved.yaml"), "code")


if __name__ == "__main__":
    main()
