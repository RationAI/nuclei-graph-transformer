r"""Q2: shape vs texture vs position only, paired slide-level bootstrap on PANDA Radboud test.

Inference outputs only: the nucleus-level test predictions stored in the evaluation runs
(`<run>/artifacts/predictions/<slide_id>.parquet`, columns `id`, `nuclei_prediction`) of
five configurations with three seeds each (`configurations` in the config). No training,
no inference; the evaluation code is not changed.

Labels are those of the existing evaluation (`postprocessing.crop.metrics.load_and_merge_data`):
the predictions of the test slides are left-joined with the annotation supervision
(`annot_label`) on (slide_id, id), missing labels become 0, and every nucleus of a slide with
`is_carcinoma = False` is 0. No nucleus is excluded: the Radboud annotation supervision is
dense (labels 0/1 only, no ignore list). Nucleus ids (64 hex digits) are keyed by their
first 16 hex digits (64 bits); uniqueness within each slide is asserted.

Checks: the runs must cover the same nuclei (otherwise everything is evaluated on the
intersection and the drops are counted); each run's AUROC/AUPRC on its own predictions is
recomputed (sklearn, the same definitions as torchmetrics' BinaryAUROC and
BinaryAveragePrecision) and compared with the metrics logged on the run, where there are any.

Metrics: AUPRC (average precision) and AUROC over all test nuclei pooled across slides.

Bootstrap: `n_resamples` resamples of the test slides, drawn with replacement separately
among positive slides (at least one nucleus labelled carcinoma) and negative slides, made
once with a fixed seed and saved as a (resamples x slides) matrix of slide multiplicities
(`resamples_file`); later analyses reuse it. Per replicate, every run's metric weights each
nucleus by its slide's multiplicity, and the three seeds of a configuration are averaged.
Point estimate: the seed-averaged exact metric on the full test set; CI: percentiles of the
replicates. Replicates are computed from per-slide histograms of the predicted logits of
positive and negative nuclei on a fixed uniform logit grid (within-bin nuclei count as tied);
the grid is refined until every run's full-test-set AUROC and AUPRC from the histograms match
the exact values within `histogram.tolerance`.

Paired comparisons: weighted sums of the seed-averaged metrics, within each replicate. p is
2 x the smaller fraction of replicates on either side of 0 (<= 0 and >= 0), capped at 1;
Holm across the comparisons, per metric.

Outputs in `out_dir`: `configuration_<name>.csv` (one per configuration) and
`configurations.csv` (all), `comparisons.csv`, `counts.csv`, `runs.csv`, `checks.json`,
`tab_res_sources.tex`, `figures/forest_auprc.{pdf,png}`, the resamples and `replicates.npz`
(every run's replicate metrics). All are logged to one MLflow run.
"""

import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm


METRICS = ("AUPRC", "AUROC")
LOGGED_KEYS = {m: f"test_thresholded/nuclei_{m}" for m in METRICS}
THESIS_RC = {  # as xai.concept_morphometry
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 8,
    "pdf.fonttype": 42,
}
FIG_W_CM = 16.0
INK, MUTED, MARK = "#1f1f1f", "#8a8a8a", "#2a5c8f"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def nucleus_keys(ids: pd.Series) -> NDArray[np.uint64]:
    """The first 16 hex digits (64 bits) of each 64-hex-digit nucleus id."""
    assert (ids.str.len() == 64).all(), "unexpected nucleus id format"
    prefix = "".join(ids.str.slice(0, 16))
    return np.frombuffer(bytes.fromhex(prefix), dtype=">u8").astype(np.uint64)


def load_run(
    run_id: str, slide_ids: list[str], cache_dir: Path, artifact: str
) -> tuple[pd.DataFrame, dict]:
    """(slide, key, p) of the run's test predictions, sorted; cached per run."""
    path, info_path = cache_dir / f"{run_id}.parquet", cache_dir / f"{run_id}.json"
    if path.exists() and info_path.exists():
        return pd.read_parquet(path), json.loads(info_path.read_text())

    cache_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache_dir) as tmp:
        local = Path(
            download_artifacts(run_id=run_id, artifact_path=artifact, dst_path=tmp)
        )
        files = {f.stem for f in local.glob("*.parquet")}
        frames = []
        for s, slide_id in enumerate(tqdm(slide_ids, desc=f"reading {run_id[:8]}")):
            if slide_id not in files:
                continue
            d = pd.read_parquet(
                local / f"{slide_id}.parquet", columns=["id", "nuclei_prediction"]
            )
            keys = nucleus_keys(d["id"])
            assert d["id"].is_unique, f"{run_id}: duplicate nucleus ids in {slide_id}"
            assert len(np.unique(keys)) == len(keys), (
                f"64-bit key collision in {slide_id}"
            )
            frames.append(
                pd.DataFrame(
                    {
                        "slide": np.full(len(d), s, np.int16),
                        "key": keys,
                        "p": d["nuclei_prediction"].to_numpy(np.float32),
                    }
                )
            )
    df = pd.concat(frames, ignore_index=True).sort_values(
        ["slide", "key"], ignore_index=True
    )
    info = {
        "prediction_files": len(files),
        "files_outside_test_metadata": len(files - set(slide_ids)),
        "test_slides_without_predictions": len(set(slide_ids) - files),
        "nuclei": len(df),
    }
    df.to_parquet(path, index=False)
    info_path.write_text(json.dumps(info, indent=1))
    return df, info


def load_labels(
    slides: pd.DataFrame, supervision_dir: Path, label_column: str
) -> pd.DataFrame:
    """(slide, key, label) of the annotation supervision of the carcinoma test slides."""
    frames = []
    for s, row in enumerate(slides.itertuples()):
        if not row.is_carcinoma:  # every nucleus is 0, whatever the supervision says
            continue
        d = pd.read_parquet(
            supervision_dir / f"{row.slide_id}.parquet", columns=["id", label_column]
        )
        assert set(d[label_column].unique()) <= {0, 1}
        frames.append(
            pd.DataFrame(
                {
                    "slide": np.full(len(d), s, np.int16),
                    "key": nucleus_keys(d["id"]),
                    "label_sup": d[label_column].to_numpy(np.int8),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def label_predictions(
    df: pd.DataFrame, labels: pd.DataFrame, carcinoma: NDArray[np.bool_]
) -> tuple[NDArray[np.int8], int]:
    """Labels of `df`'s nuclei as in the existing evaluation, and a count.

    The count is of nuclei of carcinoma slides without a supervision row (0 by the rule).
    """
    merged = df[["slide", "key"]].merge(
        labels, on=["slide", "key"], how="left", validate="1:1"
    )
    missing = (
        merged["label_sup"].isna().to_numpy() & carcinoma[merged["slide"].to_numpy()]
    )
    return merged["label_sup"].fillna(0).to_numpy(np.int8), int(missing.sum())


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def exact_metrics(y: NDArray, p: NDArray) -> dict[str, float]:
    return {
        "AUPRC": float(average_precision_score(y, p)),
        "AUROC": float(roc_auc_score(y, p)),
    }


def to_logits(p: NDArray[np.float32]) -> NDArray[np.float64]:
    p64 = p.astype(np.float64)
    with np.errstate(divide="ignore"):
        return np.log(p64) - np.log1p(-p64)


def bin_logits(logit: NDArray, lo: float, hi: float, n_bins: int) -> NDArray[np.int64]:
    scaled = (np.clip(logit, lo, hi) - lo) / (hi - lo) * n_bins
    return np.minimum(scaled.astype(np.int64), n_bins - 1)


def slide_histograms(
    slide: NDArray, bins: NDArray, y: NDArray, n_slides: int, n_bins: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Per-slide counts of positive and negative nuclei per logit bin, (slides, bins) each."""
    flat = slide.astype(np.int64) * n_bins + bins
    size = n_slides * n_bins
    pos = np.bincount(flat[y == 1], minlength=size).reshape(n_slides, n_bins)
    neg = np.bincount(flat[y == 0], minlength=size).reshape(n_slides, n_bins)
    return pos.astype(np.float64), neg.astype(np.float64)


def hist_metrics(pos: NDArray, neg: NDArray) -> dict[str, NDArray]:
    """AUROC and AUPRC from (..., bins) weighted counts in ascending score order.

    Nuclei in one bin are tied, as in sklearn's ties (AUROC counts half, average precision
    steps once per distinct score).
    """
    n_pos, n_neg = pos.sum(-1), neg.sum(-1)
    neg_below = np.cumsum(neg, axis=-1) - neg
    auroc = (pos * (neg_below + 0.5 * neg)).sum(-1) / (n_pos * n_neg)
    pos_d, neg_d = pos[..., ::-1], neg[..., ::-1]
    tp, fp = np.cumsum(pos_d, axis=-1), np.cumsum(neg_d, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
    auprc = (pos_d * precision).sum(-1) / n_pos
    return {"AUPRC": auprc, "AUROC": auroc}


def replicate_metrics(
    pos: NDArray, neg: NDArray, weights: NDArray, chunk: int = 250
) -> dict[str, NDArray]:
    """Each replicate's metrics, nuclei weighted by their slide's multiplicity."""
    out: dict[str, list[NDArray]] = {m: [] for m in METRICS}
    for start in range(0, len(weights), chunk):
        w = weights[start : start + chunk]
        for metric, values in hist_metrics(w @ pos, w @ neg).items():
            out[metric].append(values)
    return {m: np.concatenate(v) for m, v in out.items()}


def stratified_resamples(
    positive: NDArray[np.bool_], n_resamples: int, seed: int
) -> NDArray[np.int16]:
    """(resamples, slides) multiplicities; positives then negatives are drawn per resample."""
    rng = np.random.default_rng(seed)
    multiplicities = np.zeros((n_resamples, len(positive)), np.int16)
    for stratum in (np.flatnonzero(positive), np.flatnonzero(~positive)):
        draws = rng.integers(0, len(stratum), size=(n_resamples, len(stratum)))
        for b in range(n_resamples):
            multiplicities[b, stratum] = np.bincount(draws[b], minlength=len(stratum))
    assert (multiplicities[:, positive].sum(1) == positive.sum()).all()
    assert (multiplicities[:, ~positive].sum(1) == (~positive).sum()).all()
    return multiplicities


def bootstrap_p(diff: NDArray) -> float:
    return float(min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))))


def holm(p: NDArray) -> NDArray:
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------


def fmt_p(p: float, n_resamples: int) -> str:
    """P = 0 is shown below the smallest non-zero two-sided p, 2 / n_resamples."""
    return f"<{2 / n_resamples:.3f}" if p == 0 else f"{p:.4f}"


def forest_figure(comp: pd.DataFrame, ci_level: float) -> plt.Figure:
    rows = comp[comp["metric"] == "AUPRC"].reset_index(drop=True)
    with plt.rc_context(THESIS_RC):
        fig, ax = plt.subplots(figsize=(FIG_W_CM / 2.54, 0.62 * len(rows) / 2.54 + 1.0))
        y = np.arange(len(rows))[::-1]
        ax.axvline(0, color=MUTED, lw=0.8, ls=(0, (3, 2)), zorder=1)
        for yi, r in zip(y, rows.itertuples(), strict=True):
            ax.errorbar(
                r.difference,
                yi,
                xerr=[[r.difference - r.ci_lo], [r.ci_hi - r.difference]],
                fmt="o",
                ms=3.5,
                mew=1.0,
                elinewidth=1.2,
                capsize=2.5,
                capthick=1.2,
                zorder=3,
                color=MARK,
                mfc=MARK if r.significant else "white",
            )
        ax.set_yticks(y, rows["label"])
        ax.tick_params(axis="y", length=0)
        ax.set_xlabel(
            rf"$\Delta$AUPRC (seed-averaged, {ci_level:.0%} paired bootstrap CI)"
        )
        ax.grid(axis="x", color="#e6e6e6", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        text_x = 1.02
        ax.text(
            text_x,
            1.0,
            r"$\Delta$ [CI]   $p_\mathrm{Holm}$",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            color=INK,
            fontsize=7,
        )
        for yi, r in zip(y, rows.itertuples(), strict=True):
            p = "<0.001" if r.p_holm < 0.001 else f"{r.p_holm:.3f}"
            ax.text(
                text_x,
                yi,
                f"{r.difference:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]   {p}",
                transform=ax.get_yaxis_transform(),
                ha="left",
                va="center",
                color=INK,
                fontsize=7,
            )
        ax.plot(
            [], [], "o", color=MARK, mfc=MARK, ms=3.5, label="Holm-adjusted $p < 0.05$"
        )
        ax.plot([], [], "o", color=MARK, mfc="white", ms=3.5, label="not significant")
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=2,
            frameon=False,
            fontsize=7,
        )
        fig.subplots_adjust(left=0.36, right=0.68, top=0.93, bottom=0.3)
    return fig


def latex_table(conf: pd.DataFrame, comp: pd.DataFrame, config: DictConfig) -> str:
    def point(name: str, metric: str) -> str:
        r = conf[(conf["configuration"] == name) & (conf["metric"] == metric)]
        return f"{r['point'].item():.3f}" if len(r) else "--"

    gains = {"Shape": "context_gain_shape", "Texture": "context_gain_texture"}
    rows = []
    for source, self_name, ctx_name in (
        ("Shape", "shape_self", "shape_knn"),
        ("Texture", "texture_self", "texture_knn"),
        ("Position only", None, "position_knn"),
    ):
        if source in gains:
            g = comp[
                (comp["comparison"] == gains[source]) & (comp["metric"] == "AUPRC")
            ].iloc[0]
            gain = f"{g['difference']:+.3f} [{g['ci_lo']:+.3f}, {g['ci_hi']:+.3f}]"
        else:
            gain = "--"
        cells = [
            point(self_name, "AUPRC") if self_name else "--",
            point(self_name, "AUROC") if self_name else "--",
            point(ctx_name, "AUPRC"),
            point(ctx_name, "AUROC"),
            gain,
        ]
        rows.append(f"    {source} & " + " & ".join(cells) + r" \\")
    b = config.bootstrap
    return "\n".join(
        [
            "% Generated by postprocessing/bootstrap_sources.py: seed-averaged (3 seeds) nucleus-level",
            "% metrics on the PANDA Radboud test set; context = k-NN (k = 24) + RoVE; Delta AUPRC = context gain",
            f"% with a {b.ci_level:.0%} paired bootstrap CI ({b.n_resamples} slide resamples, stratified by slide label).",
            r"\begin{table}[t]",
            r"  \centering",
            r"  \caption{Nucleus-level carcinoma detection on the PANDA Radboud test set by input source,"
            r" without and with spatial context (k-NN attention, $k = 24$, RoVE). Seed-averaged over three seeds;"
            r" $\Delta$AUPRC is the context gain with its "
            + f"{b.ci_level:.0%}".replace("%", r"\%")
            + r" paired slide-level bootstrap CI.}",
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


def reference_logits_check(
    ref: DictConfig, df: pd.DataFrame, y: NDArray, slides: pd.DataFrame
) -> dict:
    """The XAI chapter's stored test logits of the same model vs its evaluation run."""
    index = pd.read_parquet(ref.index, columns=["slide_id", "nucleus_id", "label"])
    logits = np.load(ref.logits)
    slide_of = {s: i for i, s in enumerate(slides["slide_id"])}
    xai = pd.DataFrame(
        {
            "slide": index["slide_id"].map(slide_of).to_numpy(np.int16),
            "key": nucleus_keys(index["nucleus_id"]),
            "label_xai": index["label"].to_numpy(np.int8),
            "logit_xai": logits,
        }
    )
    run = df.assign(label=y)
    m = run.merge(xai, on=["slide", "key"], how="inner", validate="1:1")
    p_xai = 1 / (1 + np.exp(-m["logit_xai"].to_numpy(np.float64)))
    return {
        "eval_run": ref.eval_run,
        "model_run": ref.model_run,
        "quoted": dict(ref.quoted),
        "xai_nuclei": len(xai),
        "eval_nuclei": len(run),
        "matched_nuclei": len(m),
        "labels_agree": bool((m["label"] == m["label_xai"]).all()),
        "xai_logits_on_xai_index": exact_metrics(xai["label_xai"], xai["logit_xai"]),
        "eval_run_on_matched": exact_metrics(m["label"], m["p"]),
        "max_abs_probability_difference": float(np.abs(p_xai - m["p"]).max()),
        "mean_abs_probability_difference": float(np.abs(p_xai - m["p"]).mean()),
        "pearson_probability": float(np.corrcoef(p_xai, m["p"])[0, 1]),
        "same_side_of_0.5": float(np.mean((p_xai >= 0.5) == (m["p"] >= 0.5))),
    }


def compute(config: DictConfig, client: MlflowClient) -> tuple[Path, dict]:
    out = Path(config.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    cache = Path(config.cache_dir)
    b = config.bootstrap

    slides = pd.read_parquet(download_artifacts(config.metadata_uri))[
        ["slide_id", "is_carcinoma"]
    ].sort_values("slide_id", ignore_index=True)
    slide_ids = slides["slide_id"].tolist()
    assert slides["slide_id"].is_unique
    labels = load_labels(slides, Path(config.supervision_dir), config.label_column)
    carcinoma = slides["is_carcinoma"].to_numpy(bool)

    names = list(config.configurations)
    runs = [
        (n, i, r)
        for n in names
        for i, r in enumerate(config.configurations[n].eval_runs)
    ]
    assert len({r for _, _, r in runs}) == len(runs), (
        "an evaluation run is listed twice"
    )

    # 1. data and checks ------------------------------------------------------
    frames, run_rows = [], []
    for name, seed_i, run_id in runs:
        df, info = load_run(run_id, slide_ids, cache, config.predictions_artifact)
        y, missing = label_predictions(df, labels, carcinoma)
        mlrun = client.get_run(run_id)
        parent = mlrun.data.tags.get("mlflow.parentRunId")
        logged = {m: mlrun.data.metrics.get(k) for m, k in LOGGED_KEYS.items()}
        own = exact_metrics(y, df["p"].to_numpy())
        run_rows.append(
            {
                "configuration": name,
                "seed": seed_i + 1,
                "eval_run": run_id,
                "training_run": parent,
                "training_run_name": client.get_run(parent).data.tags.get(
                    "mlflow.runName"
                )
                if parent
                else None,
                **info,
                "carcinoma_slide_nuclei_without_supervision_row": missing,
                **{f"{m}_own": v for m, v in own.items()},
                **{f"{m}_logged": logged[m] for m in METRICS},
                **{
                    f"{m}_own_minus_logged": (own[m] - logged[m])
                    if logged[m] is not None
                    else None
                    for m in METRICS
                },
            }
        )
        frames.append(df)
        print(
            f"{name} seed {seed_i + 1} {run_id[:8]}: {len(df):,} nuclei, own {own}, logged {logged}"
        )

    ref = frames[0][["slide", "key"]]
    same = [
        len(f) == len(ref)
        and np.array_equal(f["slide"].to_numpy(), ref["slide"].to_numpy())
        and np.array_equal(f["key"].to_numpy(), ref["key"].to_numpy())
        for f in frames
    ]
    common = ref
    if not all(same):
        for f, s in zip(frames, same, strict=True):
            if not s:
                common = common.merge(
                    f[["slide", "key"]], on=["slide", "key"], how="inner"
                )
        common = common.sort_values(["slide", "key"], ignore_index=True)
    for row, f in zip(run_rows, frames, strict=True):
        row["nuclei_dropped_by_intersection"] = len(f) - len(common)
        row["slides_with_dropped_nuclei"] = int(
            (
                f.groupby("slide").size().reindex(range(len(slides)), fill_value=0)
                - common.groupby("slide")
                .size()
                .reindex(range(len(slides)), fill_value=0)
            )
            .ne(0)
            .sum()
        )

    aligned = []
    for f, s in zip(frames, same, strict=True):
        aligned.append(
            f["p"].to_numpy()
            if s and len(common) == len(f)
            else common.merge(f, on=["slide", "key"], how="left")["p"].to_numpy()
        )
    probs = np.stack(aligned)  # (runs, nuclei)
    y, missing_common = label_predictions(common, labels, carcinoma)
    slide = common["slide"].to_numpy()
    n_slides = len(slides)

    pos_per_slide = np.bincount(slide, weights=(y == 1), minlength=n_slides)
    nuc_per_slide = np.bincount(slide, minlength=n_slides)
    positive = pos_per_slide > 0
    counts = {
        "test_slides": n_slides,
        "slides_with_nuclei": int((nuc_per_slide > 0).sum()),
        "positive_slides": int(positive.sum()),
        "negative_slides": int((~positive).sum()),
        "is_carcinoma_slides": int(carcinoma.sum()),
        "is_carcinoma_slides_without_carcinoma_nuclei": int(
            (carcinoma & ~positive).sum()
        ),
        "nuclei": len(y),
        "carcinoma_nuclei": int((y == 1).sum()),
        "benign_nuclei": int((y == 0).sum()),
        "prevalence": float(np.mean(y == 1)),
        "nuclei_on_positive_slides": int(nuc_per_slide[positive].sum()),
        "prevalence_on_positive_slides": float(
            pos_per_slide[positive].sum() / nuc_per_slide[positive].sum()
        ),
        "carcinoma_slide_nuclei_without_supervision_row": missing_common,
        "runs_covering_identical_nuclei": bool(all(same)),
        "nuclei_dropped_by_intersection_max": int(
            max(r["nuclei_dropped_by_intersection"] for r in run_rows)
        ),
    }
    assert (nuc_per_slide > 0).all(), "a test slide has no nuclei"
    print(json.dumps(counts, indent=1))

    exact = [exact_metrics(y, p) for p in probs]
    for row, e in zip(run_rows, exact, strict=True):
        row.update({f"{m}_common": e[m] for m in METRICS})

    # 3. histograms, validated against the exact metrics ----------------------
    logit_rows = [to_logits(p) for p in probs]
    finite = np.concatenate([lg[np.isfinite(lg)] for lg in logit_rows])
    lo, hi = float(finite.min()), float(finite.max())
    del finite
    n_bins = int(config.histogram.n_bins)
    while True:
        hists, errors = [], []
        for lg, e in zip(logit_rows, exact, strict=True):
            pos, neg = slide_histograms(
                slide, bin_logits(lg, lo, hi, n_bins), y, n_slides, n_bins
            )
            full = hist_metrics(pos.sum(0), neg.sum(0))
            errors.append({m: float(full[m] - e[m]) for m in METRICS})
            hists.append((pos, neg))
        max_err = max(abs(v) for err in errors for v in err.values())
        print(
            f"histogram grid {n_bins} bins over logit [{lo:.3f}, {hi:.3f}]: max |error| {max_err:.2e}"
        )
        if max_err <= config.histogram.tolerance or n_bins >= config.histogram.max_bins:
            break
        n_bins *= 2
    assert max_err <= config.histogram.tolerance, (
        "histogram grid too coarse at max_bins"
    )
    for row, err in zip(run_rows, errors, strict=True):
        row.update({f"{m}_hist_minus_exact": err[m] for m in METRICS})

    multiplicities = stratified_resamples(positive, int(b.n_resamples), int(b.seed))
    np.savez_compressed(
        out / config.resamples_file,
        slide_ids=np.array(slide_ids),
        positive=positive,
        multiplicities=multiplicities,
        seed=int(b.seed),
        rule=np.array(
            "positive slide = at least one test nucleus labelled carcinoma (annot_label = 1 on an "
            "is_carcinoma slide); resampled with replacement separately among positive and "
            "negative slides; rows = resamples, columns = slide_ids"
        ),
    )

    weights = multiplicities.astype(np.float64)
    rep = {m: np.empty((len(runs), len(weights))) for m in METRICS}
    for i, (pos, neg) in enumerate(tqdm(hists, desc="bootstrap replicates")):
        for m, v in replicate_metrics(pos, neg, weights).items():
            rep[m][i] = v
    del hists
    np.savez_compressed(
        out / "replicates.npz",
        eval_runs=np.array([r for _, _, r in runs]),
        configurations=np.array([n for n, _, _ in runs]),
        **{m: rep[m] for m in METRICS},
    )

    # 3b. configurations ------------------------------------------------------
    alpha_ci = (1 - b.ci_level) / 2 * 100
    seed_avg_rep: dict[str, dict[str, NDArray]] = {m: {} for m in METRICS}
    seed_avg_point: dict[str, dict[str, float]] = {m: {} for m in METRICS}
    conf_rows = []
    for name in names:
        idx = [i for i, (n, _, _) in enumerate(runs) if n == name]
        c = config.configurations[name]
        for m in METRICS:
            per_seed = np.array([exact[i][m] for i in idx])
            seed_avg_rep[m][name] = rep[m][idx].mean(0)
            seed_avg_point[m][name] = float(per_seed.mean())
            lo_ci, hi_ci = np.percentile(
                seed_avg_rep[m][name], [alpha_ci, 100 - alpha_ci]
            )
            conf_rows.append(
                {
                    "configuration": name,
                    "label": c.label,
                    "source": c.source,
                    "context": c.context,
                    "metric": m,
                    "point": seed_avg_point[m][name],
                    "ci_lo": float(lo_ci),
                    "ci_hi": float(hi_ci),
                    "seed_sd": float(per_seed.std(ddof=1)),
                    **{f"seed_{k + 1}": float(v) for k, v in enumerate(per_seed)},
                    **{f"run_{k + 1}": runs[i][2] for k, i in enumerate(idx)},
                }
            )
    conf = pd.DataFrame(conf_rows)
    conf.to_csv(out / "configurations.csv", index=False)
    for name in names:
        conf[conf["configuration"] == name].to_csv(
            out / f"configuration_{name}.csv", index=False
        )

    # 4. paired comparisons ---------------------------------------------------
    comp_rows = []
    for m in METRICS:
        rows = []
        for c in config.comparisons:
            diff = sum(w * seed_avg_rep[m][n] for n, w in c.weights.items())
            point = sum(w * seed_avg_point[m][n] for n, w in c.weights.items())
            lo_ci, hi_ci = np.percentile(diff, [alpha_ci, 100 - alpha_ci])
            rows.append(
                {
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
        adjusted = holm(np.array([r["p"] for r in rows]))
        for r, a in zip(rows, adjusted, strict=True):
            r["p_holm"] = float(a)
            r["significant"] = bool(a < b.alpha)
            r["p_display"] = fmt_p(r["p"], int(b.n_resamples))
            r["p_holm_display"] = fmt_p(float(a), int(b.n_resamples))
        comp_rows += rows
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(out / "comparisons.csv", index=False)

    # 5. outputs --------------------------------------------------------------
    pd.DataFrame([counts]).to_csv(out / "counts.csv", index=False)
    runs_df = pd.DataFrame(run_rows)
    runs_df.to_csv(out / "runs.csv", index=False)
    (out / "tab_res_sources.tex").write_text(latex_table(conf, comp, config))
    fig = forest_figure(comp, float(b.ci_level))
    fig.savefig(out / "figures/forest_auprc.pdf", bbox_inches="tight")
    fig.savefig(out / "figures/forest_auprc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    ref_idx = next(
        i for i, (_, _, r) in enumerate(runs) if r == config.reference_logits.eval_run
    )
    ref_check = reference_logits_check(
        config.reference_logits,
        frames[ref_idx],
        label_predictions(frames[ref_idx], labels, carcinoma)[0],
        slides,
    )
    logged = runs_df.dropna(subset=["AUPRC_logged"])
    checks = {
        "counts": counts,
        "labels": "annot_label left-joined on (slide_id, id), missing -> 0, is_carcinoma = False -> 0 "
        "(postprocessing.crop.metrics.load_and_merge_data); no nucleus excluded",
        "positive_slide_rule": "at least one nucleus labelled carcinoma",
        "runs_covering_identical_nuclei": counts["runs_covering_identical_nuclei"],
        "logged_metrics_reproduced": {
            r.eval_run: {
                m: {
                    "logged": getattr(r, f"{m}_logged"),
                    "recomputed": getattr(r, f"{m}_own"),
                }
                for m in METRICS
            }
            for r in logged.itertuples()
        },
        "logged_max_abs_difference": float(
            np.abs(
                logged[[f"{m}_own_minus_logged" for m in METRICS]].to_numpy(float)
            ).max()
        )
        if len(logged)
        else None,
        "runs_without_logged_test_metrics": runs_df.loc[
            runs_df["AUPRC_logged"].isna(), "eval_run"
        ].tolist(),
        "histogram": {
            "n_bins": n_bins,
            "logit_range": [lo, hi],
            "tolerance": float(config.histogram.tolerance),
            "max_abs_error": max_err,
        },
        "bootstrap": {
            "n_resamples": int(b.n_resamples),
            "seed": int(b.seed),
            "ci_level": float(b.ci_level),
            "positive_slides": int(positive.sum()),
            "negative_slides": int((~positive).sum()),
        },
        "reference_logits": ref_check,
    }
    (out / "checks.json").write_text(json.dumps(checks, indent=1))
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
            "postprocessing/bootstrap_sources.py",
            "configs/postprocessing/bootstrap_sources.yaml",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return f"{commit}{' (dirty)' if dirty else ''}"


@with_cli_args(["+postprocessing=bootstrap_sources"])
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
        for key in ("difference", "ci_lo", "ci_hi", "p", "p_holm"):
            metrics[f"diff/{r.comparison}/{r.metric}_{key}"] = getattr(r, key)
    logger.log_metrics(metrics)

    for name in (
        *sorted(p.name for p in out.glob("*.csv")),
        "checks.json",
        "tab_res_sources.tex",
        "replicates.npz",
        config.resamples_file,
    ):
        logger.log_artifact(str(out / name))
    for name in ("forest_auprc.pdf", "forest_auprc.png"):
        logger.log_artifact(str(out / "figures" / name), "figures")
    for source in (__file__, "configs/postprocessing/bootstrap_sources.yaml"):
        logger.log_artifact(str(Path(__file__).parents[1] / source), "code")
    (out / "config-resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    logger.log_artifact(str(out / "config-resolved.yaml"), "code")


if __name__ == "__main__":
    main()
