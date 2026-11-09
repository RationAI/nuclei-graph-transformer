r"""Q2 (information sources): final figure and table, plus preliminary Q3 comparisons.

PANDA Radboud test set, nucleus level. Inference outputs only: the test predictions stored in
the evaluation runs (`<run>/artifacts/predictions/<slide_id>.parquet`) of the configurations
in the config, three seeds each. No training, no inference; the evaluation code is not changed.

Everything follows `postprocessing.bootstrap_sources` (the Q2 run) and reuses its helpers: the
same labels, nucleus keys and metrics, the slide resamples read from the Q2 run's file (not
redrawn; its slides and strata must equal the ones recomputed here), and the Q2 run's logit grid,
so that the shared configurations reproduce the Q2 run exactly. Replicates average the seeds of
a configuration within each resample; the point estimate is the seed-averaged exact metric.

Comparisons: weighted sums of seed-averaged metrics within each replicate, p = 2 x the smaller
fraction of replicates on either side of 0. Holm per metric within a family with `holm: true`.
The preliminary families have no Holm correction yet: p_holm is empty, and `significant` says so.

Checks, per evaluation run (`checks.csv`): the nuclei covered; test AUROC/AUPRC recomputed from
the predictions and compared with the logged ones, where logged; and the checkpoint used,
compared with the best epoch of the metric the training run's ModelCheckpoint monitored
(`monitor`, `checkpoint_mode` of its resolved config). Old evaluation configs do not record the
checkpoint; the training run's only checkpoint artifact (`save_top_k: 1`) is then taken as it,
and `checkpoint_source` says so.

Precision-recall curves: from each run's full-test-set logit histogram, one operating point per
bin with nuclei (nuclei in a bin are tied); per distinct recall the last operating point is
kept, and precision is interpolated linearly onto a common recall grid. The figure shows the
seed mean and shades the seeds' min-max.

Outputs in `out_dir`: `configurations.csv`, `comparisons.csv`, `checks.csv`, `counts.csv`,
`checks.json`, `pr_curves.npz`, `replicates.npz`, `tab_res_sources.tex` and
`figures/res_sources.{pdf,png}`. All are logged to one MLflow run, with the code and the
resolved config.
"""

import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib.gridspec import GridSpec
from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from tqdm import tqdm

from postprocessing.bootstrap_sources import (
    LOGGED_KEYS,
    METRICS,
    THESIS_RC,
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


MUTED = "#8a8a8a"
CHECKPOINT = re.compile(r"epoch=(\d+)-step=(\d+)")


# --------------------------------------------------------------------------
# Per-run checks
# --------------------------------------------------------------------------


def date(ms: int | None) -> str | None:
    return datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%d") if ms else None


def resolved_config(run_id: str) -> dict:
    with TemporaryDirectory() as tmp:
        for name in ("configs/config-resolved.yaml", "configs/config.yaml"):
            try:
                path = download_artifacts(
                    run_id=run_id, artifact_path=name, dst_path=tmp
                )
            except MlflowException:
                continue
            return yaml.safe_load(Path(path).read_text())
    return {}


def monitored_metric(train_config: dict) -> tuple[str, str]:
    """The metric and mode of the training run's ModelCheckpoint."""
    if train_config.get("monitor"):
        return train_config["monitor"], train_config["checkpoint_mode"]
    text = json.dumps(train_config, default=str)
    return (
        re.search(r'"monitor": "([^"]+)"', text).group(1),  # type: ignore[union-attr]
        re.search(r'"mode": "(min|max)"', text).group(1),  # type: ignore[union-attr]
    )


def checkpoint_check(client: MlflowClient, run_id: str, fix_date: str) -> dict:
    """The checkpoint this evaluation used vs the training run's best validation epoch."""
    run = client.get_run(run_id)
    parent = run.data.tags["mlflow.parentRunId"]
    train = client.get_run(parent)
    train_config = resolved_config(parent)
    artifacts = [a.path for a in client.list_artifacts(parent, "checkpoints")]

    match = CHECKPOINT.search(str(resolved_config(run_id).get("checkpoint")))
    source = "evaluation config"
    if match is None and len(artifacts) == 1:
        match = CHECKPOINT.search(artifacts[0])
        source = "not recorded; the training run's only checkpoint artifact"
    epoch, step = (int(match[1]), int(match[2])) if match else (None, None)

    monitor, mode = monitored_metric(train_config)
    history = client.get_metric_history(parent, monitor)
    best = (min if mode == "min" else max)(history, key=lambda m: m.value)
    epochs = {m.step: int(m.value) for m in client.get_metric_history(parent, "epoch")}
    at_checkpoint = [m.value for m in history if abs(m.step - (step or -9)) <= 1]
    eval_date, train_date = date(run.info.start_time), date(train.info.start_time)
    return {
        "eval_run": run_id,
        "eval_date": eval_date,
        "training_run": parent,
        "training_run_name": train.info.run_name,
        "training_status": train.info.status,
        "training_date": train_date,
        "checkpoint_epoch": epoch,
        "checkpoint_step": step,
        "checkpoint_source": source,
        "monitor": monitor,
        "monitor_mode": mode,
        "best_epoch": epochs.get(best.step),
        "best_value": best.value,
        "value_at_checkpoint": at_checkpoint[0] if at_checkpoint else None,
        "last_logged_epoch": epochs.get(max(m.step for m in history)),
        "checkpoint_is_best": epoch == epochs.get(best.step),
        "training_unfinished": train.info.status != "FINISHED",
        "validation_AUROC_at_evaluation": run.data.metrics.get("validation/AUROC"),
        "validation_AUROC_best_in_training": train.data.metrics.get(
            "best/validation/AUROC"
        ),
        "evaluated_before_block_mask_fix": eval_date < fix_date,
        "trained_before_block_mask_fix": train_date < fix_date,
        "logged": {m: run.data.metrics.get(k) for m, k in LOGGED_KEYS.items()},
    }


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------


def pr_curve(pos: NDArray, neg: NDArray, grid: NDArray) -> NDArray:
    """Precision on `grid` (recall), from ascending-score histogram counts."""
    pos_d, neg_d = pos[::-1], neg[::-1]
    keep = (pos_d + neg_d) > 0
    tp, fp = np.cumsum(pos_d)[keep], np.cumsum(neg_d)[keep]
    recall, precision = tp / pos.sum(), tp / (tp + fp)
    last = np.r_[recall[1:] != recall[:-1], True]  # last operating point per recall
    recall, precision = recall[last], precision[last]
    if recall[0] > 0:
        recall, precision = np.r_[0.0, recall], np.r_[precision[0], precision]
    return np.interp(grid, recall, precision)


def sources_figure(
    config: DictConfig,
    conf: pd.DataFrame,
    curves: dict[str, NDArray],
    grid: NDArray,
    prevalence: float,
) -> plt.Figure:
    f = config.figure
    names = list(f.configurations)
    sources = [config.configurations[n].source for n in names]
    colors = [f.colors[s] for s in sources]
    labels = [s.lower() for s in sources]
    rows = conf[conf["metric"] == "AUPRC"].set_index("configuration")
    ylim = (prevalence - 0.05, 1.005)

    with plt.rc_context({**THESIS_RC, "font.size": f.font_size}):
        fig = plt.figure(figsize=(f.width_in, f.height_in))
        gs = GridSpec(
            1,
            2,
            figure=fig,
            width_ratios=[1.0, 1.25],
            wspace=0.26,
            left=0.075,
            right=0.975,
            bottom=0.17,
            top=0.93,
        )
        ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

        for x, (name, color) in enumerate(zip(names, colors, strict=True)):
            r = rows.loc[name]
            ax_a.errorbar(
                x,
                r["point"],
                yerr=[[r["point"] - r["ci_lo"]], [r["ci_hi"] - r["point"]]],
                fmt="o",
                ms=3.0,
                color=color,
                mfc=color,
                mew=0.6,
                elinewidth=0.9,
                capsize=2.2,
                capthick=0.9,
                zorder=3,
            )
            seeds = [r[f"seed_{k + 1}"] for k in range(int(r["n_seeds"]))]
            ax_a.plot(
                x + 0.2 + 0.08 * np.arange(len(seeds)),
                seeds,
                "o",
                ms=2.6,
                mfc="white",
                mec=color,
                mew=0.7,
                ls="none",
                zorder=3,
            )
        ax_a.axhline(prevalence, color=MUTED, lw=0.8, ls=(0, (4, 2)), zorder=1)
        ax_a.text(
            0.98,
            prevalence + 0.006,
            "random",
            transform=ax_a.get_yaxis_transform(),
            ha="right",
            va="bottom",
            color=MUTED,
            fontsize=f.font_size - 1,
        )
        ax_a.set_xticks(range(len(names)), labels)
        ax_a.tick_params(axis="x", length=0)
        ax_a.set_xlim(-0.5, len(names) - 0.3)
        ax_a.set_ylabel("AUPRC")

        for name, label, color in zip(names, labels, colors, strict=True):
            c = curves[name]
            ax_b.fill_between(grid, c.min(0), c.max(0), color=color, alpha=0.25, lw=0)
            ax_b.plot(grid, c.mean(0), color=color, lw=1.1, label=label)
        ax_b.set_xlim(0, 1)
        ax_b.set_xlabel("Recall")
        ax_b.set_ylabel("Precision")
        ax_b.legend(
            loc="lower left", frameon=False, fontsize=f.font_size - 1, handlelength=2.0
        )

        for ax, tag in ((ax_a, "(a)"), (ax_b, "(b)")):
            ax.set_ylim(*ylim)
            ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.text(0.0, 1.02, tag, transform=ax.transAxes, ha="left", va="bottom")
    return fig


# --------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------


def signed(x: float) -> str:
    return ("+" if x >= 0 else "$-$") + f"{abs(x):.3f}"


def plain(x: float) -> str:
    return ("$-$" if x < 0 else "") + f"{abs(x):.3f}"


def sources_table(conf: pd.DataFrame, comp: pd.DataFrame, config: DictConfig) -> str:
    def point(name: str, metric: str) -> str:
        r = conf[(conf["configuration"] == name) & (conf["metric"] == metric)]
        return f"{r['point'].item():.3f}"

    def gain(name: str) -> str:
        r = comp[
            (comp["family"] == "sources")
            & (comp["comparison"] == name)
            & (comp["metric"] == "AUPRC")
        ].iloc[0]
        mark = r"$^{*}$" if r["significant"] == "yes" else ""
        return f"{signed(r['difference'])}{mark} [{plain(r['ci_lo'])}, {plain(r['ci_hi'])}]"

    rows = []
    for source, self_name, ctx_name, gain_name in (
        ("Shape", "shape_self", "shape_knn", "context_gain_shape"),
        ("Texture", "texture_self", "texture_knn", "context_gain_texture"),
        ("Position only", None, "position_knn", None),
    ):
        cells = [
            point(self_name, "AUPRC") if self_name else "--",
            point(self_name, "AUROC") if self_name else "--",
            point(ctx_name, "AUPRC"),
            point(ctx_name, "AUROC"),
            gain(gain_name) if gain_name else "--",
        ]
        rows.append(f"    {source} & " + " & ".join(cells) + r" \\")
    ci = f"{config.bootstrap.ci_level:.0%}".replace("%", r"\%")
    return "\n".join(
        [
            "% Generated by postprocessing/results_sources.py (PANDA Radboud test set).",
            r"\begin{table}[t]",
            r"  \centering",
            r"  \caption{Nucleus-level carcinoma detection on the PANDA Radboud test set by input"
            r" source, without and with spatial context ($k$-NN attention, $k = 24$, rotated values)."
            r" Seed-averaged over three seeds; $\Delta$AUPRC is the context gain with its "
            + ci
            + r" paired slide-level bootstrap CI. $^{*}$ Holm-adjusted $p < "
            + f"{config.bootstrap.alpha:g}"
            + r"$ across the eight source comparisons.}",
            r"  \label{tab:res-sources}",
            r"  \begin{tabular}{lccccc}",
            r"    \toprule",
            r"    & \multicolumn{2}{c}{Self-only} & \multicolumn{2}{c}{With context} & \\",
            r"    \cmidrule(lr){2-3} \cmidrule(lr){4-5}",
            r"    Source & AUPRC & AUROC & AUPRC & AUROC & $\Delta$AUPRC [CI] \\",
            r"    \midrule",
            *rows,
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )


# --------------------------------------------------------------------------
# Main computation
# --------------------------------------------------------------------------


def compute(config: DictConfig, client: MlflowClient) -> tuple[Path, dict]:
    out = Path(config.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    cache = Path(config.cache_dir)
    q2_dir = Path(config.q2_dir)
    b = config.bootstrap

    slides = pd.read_parquet(download_artifacts(config.metadata_uri))[
        ["slide_id", "is_carcinoma"]
    ].sort_values("slide_id", ignore_index=True)
    slide_ids = slides["slide_id"].tolist()
    labels = load_labels(slides, Path(config.supervision_dir), config.label_column)
    carcinoma = slides["is_carcinoma"].to_numpy(bool)

    names = list(config.configurations)
    runs = [
        (n, i, r)
        for n in names
        for i, r in enumerate(config.configurations[n].eval_runs)
    ]
    assert len({r for _, _, r in runs}) == len(runs), "a run is listed twice"

    # 1. nuclei, reproduction, checkpoints --------------------------------------
    saved = np.load(config.resamples)
    reference = None
    probs_list, run_rows = [], []
    for name, seed_i, run_id in runs:
        df, info = load_run(run_id, slide_ids, cache, config.predictions_artifact)
        y, missing = label_predictions(df, labels, carcinoma)
        if reference is None:
            reference = df[["slide", "key"]]
            y_ref, slide = y, df["slide"].to_numpy()
        same = (
            len(df) == len(reference)
            and np.array_equal(df["slide"].to_numpy(), reference["slide"].to_numpy())
            and np.array_equal(df["key"].to_numpy(), reference["key"].to_numpy())
        )
        assert same, f"{run_id} does not cover the nucleus set of the first run"
        own = exact_metrics(y, df["p"].to_numpy())
        check = checkpoint_check(client, run_id, str(config.block_mask_fix_date))
        logged = check.pop("logged")
        run_rows.append(
            {
                "configuration": name,
                "seed": seed_i + 1,
                **check,
                **info,
                "same_nuclei_as_first_run": same,
                "carcinoma_slide_nuclei_without_supervision_row": missing,
                **{f"{m}_recomputed": own[m] for m in METRICS},
                **{f"{m}_logged": logged[m] for m in METRICS},
                **{
                    f"{m}_recomputed_minus_logged": (own[m] - logged[m])
                    if logged[m] is not None
                    else None
                    for m in METRICS
                },
            }
        )
        probs_list.append(df["p"].to_numpy())
        print(f"{name} seed {seed_i + 1} {run_id[:8]}: {own}", flush=True)
    probs = np.stack(probs_list)
    del probs_list
    y = y_ref
    n_slides = len(slides)

    pos_per_slide = np.bincount(slide, weights=(y == 1), minlength=n_slides)
    nuc_per_slide = np.bincount(slide, minlength=n_slides)
    positive = pos_per_slide > 0
    assert (nuc_per_slide > 0).all(), "a test slide has no nuclei"
    counts = {
        "test_slides": n_slides,
        "positive_slides": int(positive.sum()),
        "negative_slides": int((~positive).sum()),
        "nuclei": len(y),
        "carcinoma_nuclei": int((y == 1).sum()),
        "benign_nuclei": int((y == 0).sum()),
        "prevalence": float(np.mean(y == 1)),
        "runs_covering_identical_nuclei": True,
    }

    # 2. the saved resamples ----------------------------------------------------
    assert saved["slide_ids"].tolist() == slide_ids, "resamples: different slides"
    assert np.array_equal(saved["positive"], positive), "resamples: different strata"
    multiplicities = saved["multiplicities"]
    assert multiplicities.shape[1] == n_slides
    assert (multiplicities[:, positive].sum(1) == positive.sum()).all()
    assert (multiplicities[:, ~positive].sum(1) == (~positive).sum()).all()
    n_resamples = len(multiplicities)
    weights = multiplicities.astype(np.float64)

    # 3. histograms on the Q2 grid, validated, then replicates ------------------
    q2_checks = json.loads((q2_dir / "checks.json").read_text())
    assert counts["nuclei"] == q2_checks["counts"]["nuclei"], "not the Q2 nucleus set"
    assert counts["carcinoma_nuclei"] == q2_checks["counts"]["carcinoma_nuclei"]
    lo, hi = q2_checks["histogram"]["logit_range"]
    n_bins = int(q2_checks["histogram"]["n_bins"])
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
    for row, err in zip(run_rows, errors, strict=True):
        row.update({f"{m}_hist_minus_exact": err[m] for m in METRICS})

    grid = np.linspace(0, 1, int(config.figure.recall_grid))
    rep = {m: np.empty((len(runs), n_resamples)) for m in METRICS}
    curves: dict[str, list[NDArray]] = {n: [] for n in config.figure.configurations}
    for i, p in enumerate(tqdm(probs, desc="bootstrap replicates")):
        bins = bin_logits(to_logits(p), lo, hi, n_bins)
        pos, neg = slide_histograms(slide, bins, y, n_slides, n_bins)
        if runs[i][0] in curves:
            curves[runs[i][0]].append(pr_curve(pos.sum(0), neg.sum(0), grid))
        for m, v in replicate_metrics(pos, neg, weights).items():
            rep[m][i] = v
    curve_arrays = {n: np.stack(c) for n, c in curves.items()}
    np.savez_compressed(
        out / "replicates.npz",
        eval_runs=np.array([r for _, _, r in runs]),
        configurations=np.array([n for n, _, _ in runs]),
        **{m: rep[m] for m in METRICS},
    )
    np.savez_compressed(out / "pr_curves.npz", recall=grid, **curve_arrays)

    # 4. configurations ---------------------------------------------------------
    alpha_ci = (1 - b.ci_level) / 2 * 100
    avg_rep: dict[str, dict[str, NDArray]] = {m: {} for m in METRICS}
    avg_point: dict[str, dict[str, float]] = {m: {} for m in METRICS}
    conf_rows = []
    for name in names:
        idx = [i for i, (n, _, _) in enumerate(runs) if n == name]
        c = config.configurations[name]
        for m in METRICS:
            per_seed = np.array([exact[i][m] for i in idx])
            avg_rep[m][name] = rep[m][idx].mean(0)
            avg_point[m][name] = float(per_seed.mean())
            lo_ci, hi_ci = np.percentile(avg_rep[m][name], [alpha_ci, 100 - alpha_ci])
            conf_rows.append(
                {
                    "configuration": name,
                    "label": c.label,
                    "source": c.source,
                    "metric": m,
                    "n_seeds": len(idx),
                    "point": avg_point[m][name],
                    "ci_lo": float(lo_ci),
                    "ci_hi": float(hi_ci),
                    "seed_sd": float(per_seed.std(ddof=1)),
                    **{f"seed_{k + 1}": float(v) for k, v in enumerate(per_seed)},
                    **{f"run_{k + 1}": runs[i][2] for k, i in enumerate(idx)},
                }
            )
    conf = pd.DataFrame(conf_rows)
    conf.to_csv(out / "configurations.csv", index=False)

    # 5. paired comparisons -----------------------------------------------------
    comp_rows = []
    for family, spec in config.families.items():
        for m in METRICS:
            rows = []
            for c in spec.comparisons:
                diff = sum(w * avg_rep[m][n] for n, w in c.weights.items())
                point = sum(w * avg_point[m][n] for n, w in c.weights.items())
                lo_ci, hi_ci = np.percentile(diff, [alpha_ci, 100 - alpha_ci])
                rows.append(
                    {
                        "family": family,
                        "status": "final" if spec.holm else "preliminary",
                        "metric": m,
                        "comparison": c.name,
                        "label": c.label,
                        "weights": json.dumps(dict(c.weights)),
                        "difference": float(point),
                        "ci_lo": float(lo_ci),
                        "ci_hi": float(hi_ci),
                        "ci_excludes_zero": bool(lo_ci > 0 or hi_ci < 0),
                        "replicate_mean": float(diff.mean()),
                        "p": bootstrap_p(diff),
                    }
                )
            p = np.array([r["p"] for r in rows])
            adjusted = holm(p) if spec.holm else np.full(len(p), np.nan)
            for r, a in zip(rows, adjusted, strict=True):
                r["p_holm"] = float(a)
                r["significant"] = (
                    ("yes" if a < b.alpha else "no")
                    if spec.holm
                    else "not tested: Holm pending"
                )
                r["p_display"] = fmt_p(r["p"], n_resamples)
                r["p_holm_display"] = fmt_p(float(a), n_resamples) if spec.holm else ""
            comp_rows += rows
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(out / "comparisons.csv", index=False)

    # 6. outputs ----------------------------------------------------------------
    pd.DataFrame([counts]).to_csv(out / "counts.csv", index=False)
    checks_df = pd.DataFrame(run_rows)
    checks_df.to_csv(out / "checks.csv", index=False)
    (out / "tab_res_sources.tex").write_text(sources_table(conf, comp, config))
    # Fonts are resolved when drawing, so saving needs the same rc as building.
    with plt.rc_context({**THESIS_RC, "font.size": config.figure.font_size}):
        fig = sources_figure(config, conf, curve_arrays, grid, counts["prevalence"])
        fig.savefig(out / "figures/res_sources.pdf")
        fig.savefig(out / "figures/res_sources.png", dpi=300)
    plt.close(fig)

    # 7. checks summary ---------------------------------------------------------
    previous = pd.read_csv(q2_dir / "configurations.csv")
    previous_comp = pd.read_csv(q2_dir / "comparisons.csv")
    merged = previous.merge(conf, on=["configuration", "metric"], suffixes=("_q2", ""))
    merged_comp = previous_comp.merge(
        comp[comp["family"] == "sources"],
        on=["comparison", "metric"],
        suffixes=("_q2", ""),
    )
    logged = checks_df.dropna(subset=["AUPRC_logged"])
    checks = {
        "counts": counts,
        "labels": "annot_label left-joined on (slide_id, id), missing -> 0, is_carcinoma = False -> 0 "
        "(postprocessing.crop.metrics.load_and_merge_data); no nucleus excluded",
        "resamples": {
            "file": str(config.resamples),
            "n_resamples": n_resamples,
            "seed": int(saved["seed"]),
            "slides_identical": True,
            "strata_identical": True,
        },
        "reproduction": {
            "runs_with_logged_test_metrics": logged["eval_run"].tolist(),
            "runs_without_logged_test_metrics": checks_df.loc[
                checks_df["AUPRC_logged"].isna(), "eval_run"
            ].tolist(),
            "max_abs_recomputed_minus_logged": float(
                np.abs(
                    logged[[f"{m}_recomputed_minus_logged" for m in METRICS]].to_numpy(
                        float
                    )
                ).max()
            ),
        },
        "checkpoints": {
            "not_best_validation": checks_df.loc[
                ~checks_df["checkpoint_is_best"], "eval_run"
            ].tolist(),
            "not_recorded_in_evaluation": checks_df.loc[
                checks_df["checkpoint_source"] != "evaluation config", "eval_run"
            ].tolist(),
            "training_unfinished": checks_df.loc[
                checks_df["training_unfinished"],
                ["eval_run", "training_run", "checkpoint_epoch", "last_logged_epoch"],
            ].to_dict("records"),
        },
        "evaluated_before_block_mask_fix": checks_df.loc[
            checks_df["evaluated_before_block_mask_fix"], "eval_run"
        ].tolist(),
        "trained_before_block_mask_fix": checks_df.loc[
            checks_df["trained_before_block_mask_fix"], "eval_run"
        ].tolist(),
        "histogram": {
            "n_bins": n_bins,
            "logit_range_q2": [lo, hi],
            "tolerance": float(config.histogram.tolerance),
            "max_abs_error": max_err,
        },
        "q2_reproduced": {
            "third_shape_model_run_q2": "777fb4134250483bab02ba57ce484ea4",
            "max_abs_configuration_difference": float(
                np.abs(
                    merged[["point", "ci_lo", "ci_hi"]].to_numpy()
                    - merged[["point_q2", "ci_lo_q2", "ci_hi_q2"]].to_numpy()
                ).max()
            ),
            "max_abs_comparison_difference": float(
                np.abs(
                    merged_comp[["difference", "ci_lo", "ci_hi", "p"]].to_numpy()
                    - merged_comp[
                        ["difference_q2", "ci_lo_q2", "ci_hi_q2", "p_q2"]
                    ].to_numpy()
                ).max()
            ),
        },
    }
    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=str))
    return out, checks


def code_commit() -> str:
    repo = Path(__file__).parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
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
            "postprocessing/results_sources.py",
            "postprocessing/bootstrap_sources.py",
            "configs/postprocessing/results_sources.yaml",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{commit}{' (dirty)' if dirty else ''}"


@with_cli_args(["+postprocessing=results_sources"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    client = logger.experiment
    client.set_tag(logger.run_id, "code_commit", code_commit())
    for name in config.configurations:
        client.set_tag(
            logger.run_id,
            f"eval_runs.{name}",
            ",".join(config.configurations[name].eval_runs),
        )
    out, checks = compute(config, client)

    conf = pd.read_csv(out / "configurations.csv")
    comp = pd.read_csv(out / "comparisons.csv")
    metrics = {f"counts/{k}": float(v) for k, v in checks["counts"].items()}
    metrics["histogram/n_bins"] = checks["histogram"]["n_bins"]
    metrics["histogram/max_abs_error"] = checks["histogram"]["max_abs_error"]
    for r in conf.itertuples():
        for key in ("point", "ci_lo", "ci_hi", "seed_sd"):
            metrics[f"test/{r.configuration}/{r.metric}_{key}"] = getattr(r, key)
    for r in comp.itertuples():
        keys = ("difference", "ci_lo", "ci_hi", "p")
        for key in (*keys, "p_holm") if r.status == "final" else keys:
            metrics[f"diff/{r.family}/{r.comparison}/{r.metric}_{key}"] = getattr(
                r, key
            )
    logger.log_metrics(metrics)

    for name in (
        *sorted(p.name for p in out.glob("*.csv")),
        "checks.json",
        "tab_res_sources.tex",
        "replicates.npz",
        "pr_curves.npz",
    ):
        logger.log_artifact(str(out / name))
    for name in ("res_sources.pdf", "res_sources.png"):
        logger.log_artifact(str(out / "figures" / name), "figures")
    for source in (
        __file__,
        "postprocessing/bootstrap_sources.py",
        "configs/postprocessing/results_sources.yaml",
    ):
        logger.log_artifact(str(Path(__file__).parents[1] / source), "code")
    (out / "config-resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    logger.log_artifact(str(out / "config-resolved.yaml"), "code")


if __name__ == "__main__":
    main()
