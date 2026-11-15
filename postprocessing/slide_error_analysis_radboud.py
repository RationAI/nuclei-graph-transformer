# ruff: noqa: RUF001, RUF002  (Greek letters / minus signs in report text are intended)
r"""Per-slide error analysis of the nucleus-level shape model on PANDA Radboud test.

Question: on MMCI, patients with little carcinoma had carcinoma scores shifted down (low recall at
the validation operating point) while their ranking (calibrated AUPRC) stayed good. Does the same
shift with the amount of carcinoma appear per slide on Radboud, and within grade groups?

Model, test predictions and labels of run ``results_run`` (6aa01c04, tab:res-nucleus Radboud row):
shape embedding (efd), rotated values, k-NN attention (k = 24), three seeds; the stored per-nucleus
test predictions of its evaluation runs; labels ``annot_label``, missing -> 0, nuclei of
``is_carcinoma = False`` slides -> 0 (dense, nothing ignored). Positive slide = at least one
carcinoma nucleus (776), negative = none (202).

1. Thresholds, on the seed-mean probability of the validation nuclei (the training runs' fixed split
   of the training slides, ``split_train_val``, random_state = 42, same labels rule): Youden (the
   score t maximising TPR − FPR of s ≥ t over the distinct scores, the highest t on ties) and recall
   0.90 (the highest t with recall ≥ 0.90). Validation predictions: reused if stored by a child run
   of every training run, otherwise predicted with the evaluation runs' commit (git worktree,
   ``postprocessing/nucleus_inference_at_commit.py``); the reproduction is measured by re-predicting
   the test set and comparing with the stored predictions.
2. Per positive slide: nuclei, carcinoma nuclei, prevalence, ISUP grade and Gleason score (public
   PANDA ``train.csv``); AUPRC (torchmetrics, exact), calibrated AUPRC (Siblini et al., IDA 2020:
   precision TP / (TP + c·FP), c = π(1 − π₀) / (π₀(1 − π)), π the slide's prevalence; π₀ = 0.385 and
   0.275) and AUROC, per seed then the mean; recall and FPR of the seed-mean probability at both
   thresholds (as MMCI). AUPRC values of slides with fewer than ``min_carcinoma_nuclei`` carcinoma
   nuclei are flagged unreliable and left out of the AUPRC summaries. Per negative slide: FPR.
3. Summaries: (a) prevalence bins × {all, ISUP 1, ISUP ≥ 2}: slides and median [IQR] of calibrated
   AUPRC, recall and FPR (Youden); (b) Spearman ρ of prevalence with recall and with calibrated
   AUPRC, percentile bootstrap CIs over slides; (c) 5/25/50/75/95th percentiles of the seed-mean
   scores of carcinoma and benign nuclei per prevalence bin (and of the negative slides).

Outputs (``out_dir``; logged to a new run in experiment 37 tagged with ``results_run``):
``per_slide_radboud.csv``, ``prevalence_bins_radboud.csv``, ``spearman_radboud.csv``,
``score_quantiles_radboud.csv``, ``thresholds.json``, ``report.md``. Slide ids are public PANDA
image ids; every 32-hex token of the text outputs is checked to be one of them or a run id.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import with_cli_args
from scipy.stats import rankdata
from tqdm import tqdm

from nuclei_graph.data.datamodules.base import split_train_val
from postprocessing.bootstrap_sources import label_predictions, load_labels, load_run
from postprocessing.nucleus_results_mmci import CREDENTIAL_URL, exact_metrics
from postprocessing.patient_error_analysis_mmci import (
    calibrated_ap,
    calibration,
    operating_point,
    recall_threshold,
)
from postprocessing.patient_score_quantiles_mmci import youden_threshold
from postprocessing.tile.bootstrap_pixel_vs_nuclei import git_info


REPO = Path(__file__).parents[1]
CODE = (
    "postprocessing/slide_error_analysis_radboud.py",
    "postprocessing/nucleus_inference_at_commit.py",
    "configs/postprocessing/slide_error_analysis_radboud.yaml",
)
INFERENCE = "postprocessing/nucleus_inference_at_commit.py"
CKPT_EPOCH = re.compile(r"epoch=(\d+)")
HEX32 = re.compile(r"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])")
COMMIT_TAGS = ("mlflow.source.git.commit", "code_commit", "git.commit", "git_commit")
GROUPS = ("all", "ISUP 1", "ISUP >= 2")


# --------------------------------------------------------------------------
# Runs and data
# --------------------------------------------------------------------------


def remap(path: str, config: DictConfig) -> str:
    """A path of the public cluster's project folder, as mounted here."""
    remote, local = str(config.remote_project_path), str(config.local_project_path)
    return local + path[len(remote) :] if path.startswith(remote) else path


def reference(config: DictConfig, client: MlflowClient, tmp: Path) -> dict:
    """The results run's runs, per-seed AUPRC and counts; the evaluation commit."""
    runs = pd.read_csv(
        client.download_artifacts(config.results_run, "runs.csv", str(tmp))
    )
    checks = json.loads(
        Path(
            client.download_artifacts(config.results_run, "checks.json", str(tmp))
        ).read_text()
    )
    assert list(runs["training_run"]) == list(config.training_runs), (
        "training runs differ"
    )
    assert list(runs["eval_run"]) == list(config.eval_runs), "evaluation runs differ"
    commits = []
    for e, t in zip(config.eval_runs, config.training_runs, strict=True):
        tags = client.get_run(e).data.tags
        assert tags["mlflow.parentRunId"] == t
        commit = next((tags[k] for k in COMMIT_TAGS if k in tags), None)
        assert commit, f"{e}: no commit tag among {sorted(tags)}"
        commits.append(commit.split()[0])
    assert len(set(commits)) == 1, f"evaluation runs at different commits: {commits}"
    return {
        "runs": runs,
        "counts": checks["counts"],
        "commit": commits[0],
        "checkpoint_epochs": [int(v) for v in runs["checkpoint_epoch"]],
    }


def run_config(client: MlflowClient, run_id: str, tmp: Path) -> dict:
    (tmp / run_id).mkdir(parents=True, exist_ok=True)
    path = client.download_artifacts(
        run_id, "configs/config-resolved.yaml", str(tmp / run_id)
    )
    return yaml.safe_load(Path(path).read_text())


def labelled_frames(
    slides: pd.DataFrame, frames: list[pd.DataFrame], config: DictConfig
) -> tuple[NDArray, NDArray, NDArray, int]:
    """(y, slide, probs[seed]) of every nucleus (dense labels) and the unmatched count."""
    ref = frames[0][["slide", "key"]]
    for f in frames[1:]:
        assert np.array_equal(f["slide"].to_numpy(), ref["slide"].to_numpy())
        assert np.array_equal(f["key"].to_numpy(), ref["key"].to_numpy()), (
            "seeds differ"
        )
    carcinoma = slides["is_carcinoma"].to_numpy(bool)
    labels = load_labels(slides, Path(config.supervision_dir), config.label_column)
    y, unmatched = label_predictions(ref, labels, carcinoma)
    probs = np.stack([f["p"].to_numpy() for f in frames])
    return y, ref["slide"].to_numpy(), probs, unmatched


def validation_slides(config: DictConfig, cfgs: list[dict]) -> pd.DataFrame:
    keys = ("metadata", "split_size", "split_stratify_col", "split_group_col")
    assert all(
        {k: c["data"][k] for k in keys} == {k: cfgs[0]["data"][k] for k in keys}
        for c in cfgs
    )
    d0 = cfgs[0]["data"]
    train_meta = pd.read_parquet(download_artifacts(d0["metadata"]["train"]))
    _, val = split_train_val(
        train_meta, d0["split_size"], d0["split_stratify_col"], d0["split_group_col"]
    )
    val = val.sort_values("slide_id", ignore_index=True)
    val["slide_nuclei_path"] = [remap(p, config) for p in val["slide_nuclei_path"]]
    return val


def stored_validation(
    client: MlflowClient, config: DictConfig, slide_ids: list[str], tmp: Path
) -> list[pd.DataFrame] | None:
    """Validation predictions stored by a child run of every training run, if any (else None)."""
    found = []
    for t in config.training_runs:
        children = client.search_runs(
            [config.metadata.experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{t}'",
            max_results=200,
        )
        hit = None
        for child in children:
            for art in ("predictions", "nuclei_predictions"):
                names = {
                    Path(f.path).stem
                    for f in client.list_artifacts(child.info.run_id, art)
                }
                if names and set(slide_ids) <= names:
                    hit = (child.info.run_id, art)
                    break
            if hit:
                break
        if hit is None:
            return None
        found.append(hit)
    print(f"stored validation predictions: {found}")
    return [
        load_run(r, slide_ids, tmp / "stored_validation", art)[0] for r, art in found
    ]


def predict_at_commit(
    client: MlflowClient,
    config: DictConfig,
    commit: str,
    cfgs: list[dict],
    epochs: list[int],
    slides: pd.DataFrame,
    outputs: list[Path],
    tmp: Path,
) -> list[pd.DataFrame]:
    """Predicts `slides` with each training run's checkpoint and the model code of `commit`."""
    if all(p.exists() for p in outputs):
        return [pd.read_parquet(p) for p in outputs]
    ckpts = []
    for t, epoch in zip(config.training_runs, epochs, strict=True):
        files = [f.path for f in client.list_artifacts(t, "checkpoints")]
        assert len(files) == 1, f"{t}: {len(files)} checkpoints"
        assert int(CKPT_EPOCH.search(files[0]).group(1)) == epoch, (
            "checkpoint != results run's"
        )
        local = Path(client.download_artifacts(t, files[0], str(tmp / t)))
        if local.is_dir():
            (local,) = list(local.rglob("*.ckpt"))
        ckpts.append(local)
    cfg_paths = []
    for t, c in zip(config.training_runs, cfgs, strict=True):
        p = tmp / t / "config-resolved.yaml"
        p.write_text(yaml.safe_dump(c))
        cfg_paths.append(p)
    slides_path = tmp / f"{outputs[0].stem.split('_')[-1]}_slides.parquet"
    slides.to_parquet(slides_path, index=False)
    job = tmp / f"{slides_path.stem}_job.json"
    job.write_text(
        json.dumps(
            {
                "slides": str(slides_path),
                "runs": [
                    {"checkpoint": str(c), "config": str(p)}
                    for c, p in zip(ckpts, cfg_paths, strict=True)
                ],
                "outputs": [str(o) for o in outputs],
                "device": str(config.inference.device),
            }
        )
    )
    worktree = tmp / f"code_{commit[:8]}"
    if not worktree.exists():
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), commit],
            cwd=REPO,
            check=True,
            capture_output=True,
        )
    try:
        env = {**os.environ, "PYTHONPATH": str(worktree)}
        subprocess.run(
            [
                sys.executable,
                str(REPO / INFERENCE),
                "--job",
                str(job),
                "--code-root",
                str(worktree),
            ],
            cwd=worktree,
            env=env,
            check=True,
        )
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=REPO,
            check=False,
        )
    return [pd.read_parquet(p) for p in outputs]


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def calibrated_ap_safe(y: NDArray, p: NDArray, c: float) -> float:
    """``calibrated_ap``; a slide without benign nuclei has precision 1 (c = ∞)."""
    if not (y == 0).any():
        return 1.0
    return calibrated_ap(y, p, c)


def slide_bounds(slide: NDArray, n: int) -> NDArray:
    assert np.all(np.diff(slide) >= 0), "nuclei not sorted by slide"
    return np.searchsorted(slide, np.arange(n + 1))


def per_slide(
    config: DictConfig,
    slides: pd.DataFrame,
    y: NDArray,
    slide: NDArray,
    probs: NDArray,
    thresholds: dict[str, float],
) -> tuple[pd.DataFrame, float]:
    pi0s = [float(v) for v in config.pi0]
    mean = probs.astype(np.float64).mean(0)
    bounds = slide_bounds(slide, len(slides))
    rows, max_check = [], 0.0
    for s, row in enumerate(
        tqdm(slides.itertuples(), total=len(slides), desc="slides")
    ):
        sel = slice(bounds[s], bounds[s + 1])
        ys, ms = y[sel], mean[sel]
        n, n_pos = len(ys), int(ys.sum())
        out = {
            "slide_id": row.slide_id,
            "slide_type": "positive" if n_pos > 0 else "negative",
            "n_nuclei": n,
            "n_carcinoma": n_pos,
            "prevalence": n_pos / n,
            "isup_grade": row.isup_grade,
            "gleason_score": row.gleason_score,
        }
        if n_pos > 0:
            pi = n_pos / n
            seed = []
            for p in probs[:, sel]:
                m = (
                    exact_metrics(ys, p)
                    if n_pos < n
                    else {"AUPRC": 1.0, "AUROC": np.nan}
                )
                check = calibrated_ap_safe(ys, p, 1.0)
                max_check = max(max_check, abs(check - m["AUPRC"]))
                seed.append(
                    {
                        "AUPRC": m["AUPRC"],
                        **{
                            f"calibrated_AUPRC_pi0_{q}": calibrated_ap_safe(
                                ys, p, calibration(pi, q)
                            )
                            if n_pos < n
                            else 1.0
                            for q in pi0s
                        },
                        "AUROC": m["AUROC"] if n_pos < n else np.nan,
                    }
                )
            out |= pd.DataFrame(seed).mean().to_dict()
            out["AUPRC_unreliable"] = n_pos < int(config.min_carcinoma_nuclei)
        for name, t in thresholds.items():
            o = operating_point_safe(ys, ms, t)
            out[f"recall_{name}"] = o["recall"] if n_pos > 0 else np.nan
            out[f"FPR_{name}"] = o["FPR"]
        rows.append(out)
    return pd.DataFrame(rows), max_check


def operating_point_safe(y: NDArray, s: NDArray, t: float) -> dict:
    pred = s >= t
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    return {
        "recall": tp / pos if pos else np.nan,
        "FPR": fp / neg if neg else np.nan,
    }


def positive_slides(table: pd.DataFrame) -> pd.DataFrame:
    pos = table[table["slide_type"] == "positive"].copy()
    pos["AUPRC_unreliable"] = pos["AUPRC_unreliable"].astype(bool)
    return pos


def isup_group(isup: pd.Series) -> pd.Series:
    return pd.Series(
        np.select([isup == 1, isup >= 2], ["ISUP 1", "ISUP >= 2"], "other"),
        index=isup.index,
    )


def in_group(df: pd.DataFrame, group: str) -> pd.Series:
    return (
        pd.Series(True, index=df.index) if group == "all" else df["isup_group"] == group
    )


def median_iqr(v: pd.Series) -> dict[str, float]:
    v = v.dropna()
    if v.empty:
        return {"n": 0, "median": np.nan, "q1": np.nan, "q3": np.nan}
    q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
    return {"n": len(v), "median": float(med), "q1": float(q1), "q3": float(q3)}


def bin_summary(config: DictConfig, pos: pd.DataFrame) -> pd.DataFrame:
    pi0s = [float(v) for v in config.pi0]
    rows = []
    for g in GROUPS:
        for b in [*config.prevalence_bin_labels, "all bins"]:
            sel = in_group(pos, g) & (
                (pos["prevalence_bin"] == b) if b != "all bins" else True
            )
            d = pos[sel]
            rel = d[~d["AUPRC_unreliable"]]
            row = {
                "group": g,
                "prevalence_bin": b,
                "n_slides": len(d),
                "n_slides_reliable_AUPRC": len(rel),
            }
            for col, src in (
                *[(f"calibrated_AUPRC_pi0_{q}", rel) for q in pi0s],
                ("recall_youden", d),
                ("FPR_youden", d),
            ):
                m = median_iqr(src[col])
                row |= {
                    f"{col}_median": m["median"],
                    f"{col}_q1": m["q1"],
                    f"{col}_q3": m["q3"],
                }
            rows.append(row)
    return pd.DataFrame(rows)


def spearman(x: NDArray, y: NDArray) -> float:
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def correlations(config: DictConfig, pos: pd.DataFrame) -> pd.DataFrame:
    pi0s = [float(v) for v in config.pi0]
    b = config.bootstrap
    alpha = (1 - float(b.ci_level)) / 2 * 100
    targets = [
        ("recall_youden", False),
        ("recall_recall90", False),
        *[(f"calibrated_AUPRC_pi0_{q}", True) for q in pi0s],
        ("FPR_youden", False),
    ]
    rows = []
    for g in GROUPS:
        for col, reliable_only in targets:
            d = pos[in_group(pos, g)]
            if reliable_only:
                d = d[~d["AUPRC_unreliable"]]
            d = d[["prevalence", col]].dropna()
            x, v = d["prevalence"].to_numpy(), d[col].to_numpy()
            rng = np.random.default_rng(int(b.seed))
            reps = np.empty(int(b.n_resamples))
            for i in range(len(reps)):
                idx = rng.integers(0, len(x), len(x))
                reps[i] = spearman(x[idx], v[idx])
            reps = reps[np.isfinite(reps)]
            rows.append(
                {
                    "group": g,
                    "variable": col,
                    "slides": "reliable AUPRC only"
                    if reliable_only
                    else "all positive",
                    "n_slides": len(x),
                    "spearman_rho": spearman(x, v),
                    "ci_lo": float(np.percentile(reps, alpha)),
                    "ci_hi": float(np.percentile(reps, 100 - alpha)),
                    "n_resamples": len(reps),
                }
            )
    return pd.DataFrame(rows)


def score_quantiles(
    config: DictConfig,
    table: pd.DataFrame,
    y: NDArray,
    slide: NDArray,
    mean: NDArray,
) -> pd.DataFrame:
    qs = [float(q) for q in config.quantiles]
    bin_of_slide = table["prevalence_bin"].to_numpy(dtype=object)
    group_of_slide = table["isup_group"].to_numpy(dtype=object)
    positive = (table["slide_type"] == "positive").to_numpy()
    nb, ng, npos = bin_of_slide[slide], group_of_slide[slide], positive[slide]
    rows = []

    def add(group: str, b: str, cls: str, sel: NDArray, n_slides: int) -> None:
        s = mean[sel]
        rows.append(
            {
                "group": group,
                "prevalence_bin": b,
                "class": cls,
                "n_slides": n_slides,
                "n": len(s),
                **{
                    f"p{round(q * 100)}": float(v)
                    for q, v in zip(qs, np.quantile(s, qs), strict=True)
                },
            }
        )

    for g in GROUPS:
        gsel = np.ones(len(y), bool) if g == "all" else ng == g
        for b in config.prevalence_bin_labels:
            sel = gsel & npos & (nb == b)
            n_slides = int(
                (
                    positive
                    & (bin_of_slide == b)
                    & (True if g == "all" else group_of_slide == g)
                ).sum()
            )
            if n_slides == 0:
                continue
            add(g, b, "carcinoma", sel & (y == 1), n_slides)
            add(g, b, "benign", sel & (y == 0), n_slides)
    add("negative slides", "-", "benign", ~npos, int((~positive).sum()))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def compute(config: DictConfig, client: MlflowClient) -> tuple[Path, dict, set[str]]:
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp"
    tmp.mkdir(exist_ok=True)
    ref = reference(config, client, tmp)
    cfgs = [run_config(client, t, tmp) for t in config.training_runs]

    # test: stored predictions of the evaluation runs, labels as 6aa01c04
    meta = pd.read_parquet(download_artifacts(config.metadata_uri)).sort_values(
        "slide_id", ignore_index=True
    )
    meta["slide_nuclei_path"] = [remap(p, config) for p in meta["slide_nuclei_path"]]
    stored = [
        load_run(
            e,
            meta["slide_id"].tolist(),
            Path(config.test_cache_dir),
            config.predictions_artifact,
        )[0]
        for e in config.eval_runs
    ]
    y, slide, probs, unmatched = labelled_frames(meta, stored, config)
    exact = [exact_metrics(y, p) for p in probs]
    seeds_ref = [float(v) for v in ref["runs"]["AUPRC_exact"]]
    assert (
        max(abs(e["AUPRC"] - r) for e, r in zip(exact, seeds_ref, strict=True)) < 1e-6
    ), "pooled per-seed AUPRC differs from the results run"
    c = ref["counts"]
    assert (len(y), int(y.sum()), unmatched) == (
        c["labeled_nuclei"],
        c["positive_nuclei"],
        c["carcinoma_slide_nuclei_without_supervision_row"],
    )

    # validation: stored predictions if any, else predicted at the evaluation commit
    val = validation_slides(config, cfgs)
    assert not set(val["slide_id"]) & set(meta["slide_id"]), "validation slide in test"
    cache = Path(config.inference_cache_dir) / ref["commit"][:8]
    val_pred = stored_validation(client, config, val["slide_id"].tolist(), tmp)
    validation_source = "stored"
    if val_pred is None:
        validation_source = (
            f"predicted at commit {ref['commit'][:8]} ({config.inference.device})"
        )
        val_pred = predict_at_commit(
            client,
            config,
            ref["commit"],
            cfgs,
            ref["checkpoint_epochs"],
            val,
            [cache / f"{t}_validation.parquet" for t in config.training_runs],
            tmp,
        )
    vy, _, vprobs, val_unmatched = labelled_frames(val, val_pred, config)

    rerun = None
    if config.inference.rerun_test:
        test_rerun = predict_at_commit(
            client,
            config,
            ref["commit"],
            cfgs,
            ref["checkpoint_epochs"],
            meta,
            [cache / f"{t}_test.parquet" for t in config.training_runs],
            tmp,
        )
        _, _, rprobs, _ = labelled_frames(meta, test_rerun, config)
        diff = np.abs(rprobs - probs)
        rerun = {
            "per_seed_abs_diff": [
                {
                    "mean": float(d.mean()),
                    "q99": float(np.quantile(d, 0.99)),
                    "max": float(d.max()),
                }
                for d in diff
            ],
            "pooled_AUPRC_rerun": [exact_metrics(y, p)["AUPRC"] for p in rprobs],
            "pooled_AUPRC_stored": [e["AUPRC"] for e in exact],
        }
        del rprobs, diff
    shutil.rmtree(tmp, ignore_errors=True)

    # thresholds on the seed-mean validation probability
    vmean = vprobs.astype(np.float64).mean(0)
    t90 = recall_threshold(vy, vmean, float(config.target_recall))
    t_youden, j = youden_threshold(vy, vmean)
    thresholds = {"youden": t_youden, "recall90": t90}
    threshold_info = {
        "youden": {
            "threshold": t_youden,
            "J": j,
            "rule": "argmax over distinct validation scores t of TPR - FPR (s >= t); highest t on ties",
            "validation": operating_point(vy, vmean, t_youden),
        },
        "recall90": {
            "threshold": t90,
            "rule": f"highest score with validation recall >= {config.target_recall}",
            "validation": operating_point(vy, vmean, t90),
        },
    }

    # per slide
    panda = pd.read_csv(config.panda_labels_csv)
    panda = panda[panda["data_provider"] == "radboud"].set_index("image_id")
    assert set(meta["slide_id"]) <= set(panda.index), (
        "test slide not in PANDA train.csv"
    )
    gleason = meta["slide_id"].map(panda["gleason_score"])
    if "gleason_score" in meta.columns:  # the metadata's copy of train.csv
        assert (meta["gleason_score"].astype(str) == gleason.astype(str)).all()
    meta["isup_grade"] = meta["slide_id"].map(panda["isup_grade"]).astype(int)
    meta["gleason_score"] = gleason
    table, max_check = per_slide(config, meta, y, slide, probs, thresholds)
    assert max_check <= float(config.calibration_check_tolerance), (
        f"calibrated AP with c = 1 != torchmetrics AUPRC ({max_check:.2e})"
    )
    table["isup_group"] = isup_group(table["isup_grade"])
    table["prevalence_bin"] = pd.cut(
        table["prevalence"],
        bins=[float(v) for v in config.prevalence_bins],
        labels=list(config.prevalence_bin_labels),
        right=False,
        include_lowest=True,
    ).astype(object)
    table.loc[table["prevalence"] >= 1.0, "prevalence_bin"] = (
        config.prevalence_bin_labels[-1]
    )
    table.loc[table["slide_type"] == "negative", "prevalence_bin"] = "-"
    pos = positive_slides(table)
    assert (len(pos), len(table) - len(pos)) == (
        c["positive_slides"],
        c["negative_slides"],
    )

    bins = bin_summary(config, pos)
    corr = correlations(config, pos)
    mean = probs.astype(np.float64).mean(0)
    quant = score_quantiles(config, table, y, slide, mean)

    cols = [
        "slide_id",
        "slide_type",
        "n_nuclei",
        "n_carcinoma",
        "prevalence",
        "prevalence_bin",
        "isup_grade",
        "isup_group",
        "gleason_score",
        "AUPRC",
        *[f"calibrated_AUPRC_pi0_{q}" for q in config.pi0],
        "AUROC",
        "AUPRC_unreliable",
        "recall_youden",
        "FPR_youden",
        "recall_recall90",
        "FPR_recall90",
    ]
    table[cols].to_csv(out / "per_slide_radboud.csv", index=False)
    bins.to_csv(out / "prevalence_bins_radboud.csv", index=False)
    corr.to_csv(out / "spearman_radboud.csv", index=False)
    quant.to_csv(out / "score_quantiles_radboud.csv", index=False)

    neg = table[table["slide_type"] == "negative"]
    checks = {
        "results_run": config.results_run,
        "eval_runs": list(config.eval_runs),
        "training_runs": list(config.training_runs),
        "evaluation_commit": ref["commit"],
        "checkpoint_epochs": ref["checkpoint_epochs"],
        "score": "seed-mean probability (3 seeds) for thresholds, recall, FPR and quantiles; AUPRC / calibrated AUPRC / AUROC per seed, then the mean",
        "thresholds": threshold_info,
        "test": {
            "slides": len(meta),
            "positive_slides": len(pos),
            "negative_slides": len(neg),
            "labelled_nuclei": len(y),
            "carcinoma_nuclei": int(y.sum()),
            "prevalence": float(y.mean()),
            "pooled_AUPRC_per_seed": [e["AUPRC"] for e in exact],
            "pooled_AUROC_per_seed": [e["AUROC"] for e in exact],
            "positive_slides_unreliable_AUPRC": int(pos["AUPRC_unreliable"].sum()),
            "positive_slides_without_benign_nuclei": int(
                (pos["n_carcinoma"] == pos["n_nuclei"]).sum()
            ),
            "positive_slides_by_isup_group": pos["isup_group"].value_counts().to_dict(),
            "positive_slides_isup_0": int((pos["isup_grade"] == 0).sum()),
        },
        "validation": {
            "source": validation_source,
            "slides": len(val),
            "carcinoma_slides": int(val["is_carcinoma"].sum()),
            "labelled_nuclei": len(vy),
            "carcinoma_nuclei": int(vy.sum()),
            "prevalence": float(vy.mean()),
            "carcinoma_slide_nuclei_without_supervision_row": val_unmatched,
            "split": "split_train_val(random_state=42) of the training slides",
            **{
                k: cfgs[0]["data"][k]
                for k in ("split_size", "split_stratify_col", "split_group_col")
            },
        },
        "rerun_check": rerun,
        "calibration_check": {"c": 1.0, "max_abs_diff_vs_torchmetrics": max_check},
        "rules": {
            "min_carcinoma_nuclei": int(config.min_carcinoma_nuclei),
            "prevalence_bins": list(config.prevalence_bins),
            "bootstrap": OmegaConf.to_container(config.bootstrap),
        },
    }
    (out / "thresholds.json").write_text(json.dumps(checks, indent=1, default=str))
    (out / "report.md").write_text(report(config, checks, table, bins, corr, quant))
    # public PANDA image ids, and the run ids of this analysis and of the repo's configs
    allowed = (
        set(panda.index)
        | set(config.eval_runs)
        | set(config.training_runs)
        | {config.results_run}
        | set(HEX32.findall(OmegaConf.to_yaml(config, resolve=True)))
    )
    return out, checks, allowed


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def fmt_rho(corr: pd.DataFrame, group: str, var: str) -> str:
    r = corr[(corr["group"] == group) & (corr["variable"] == var)].iloc[0]
    return (
        f"ρ = {r.spearman_rho:+.2f} [{r.ci_lo:+.2f}, {r.ci_hi:+.2f}] (n = {r.n_slides})"
    )


def rho(corr: pd.DataFrame, group: str, var: str) -> float:
    return float(
        corr[(corr["group"] == group) & (corr["variable"] == var)].iloc[0].spearman_rho
    )


def ci_sign(corr: pd.DataFrame, group: str, var: str) -> int:
    r = corr[(corr["group"] == group) & (corr["variable"] == var)].iloc[0]
    return 1 if r.ci_lo > 0 else (-1 if r.ci_hi < 0 else 0)


def report(
    config: DictConfig,
    checks: dict,
    table: pd.DataFrame,
    bins: pd.DataFrame,
    corr: pd.DataFrame,
    quant: pd.DataFrame,
) -> str:
    th, t, v = checks["thresholds"], checks["test"], checks["validation"]
    q0 = float(config.pi0[0])
    cal = f"calibrated_AUPRC_pi0_{q0}"
    pos = positive_slides(table)
    neg = table[table["slide_type"] == "negative"]
    lines = [
        "# Per-slide error analysis: PANDA Radboud test",
        "",
        (
            f"Model, test predictions and labels of run `{config.results_run}` (Shape + RoVE + k-NN attention, "
            f"k = 24, three seeds; commit `{checks['evaluation_commit'][:8]}`). {t['positive_slides']} positive and "
            f"{t['negative_slides']} negative test slides, {t['labelled_nuclei']:,} nuclei (prevalence {t['prevalence']:.4f})."
        ),
        "",
        "## Thresholds (seed-mean validation probability)",
        "",
        "| Threshold | Value | Validation recall | Validation FPR |",
        "| --- | --- | --- | --- |",
        (
            f"| Youden (max TPR − FPR, J = {th['youden']['J']:.4f}) | {th['youden']['threshold']:.4f} | "
            f"{th['youden']['validation']['recall']:.4f} | {th['youden']['validation']['FPR']:.4f} |"
        ),
        (
            f"| Recall 0.90 | {th['recall90']['threshold']:.4f} | {th['recall90']['validation']['recall']:.4f} | "
            f"{th['recall90']['validation']['FPR']:.4f} |"
        ),
        "",
        (
            f"Validation: {v['slides']} slides ({v['carcinoma_slides']} carcinoma), {v['labelled_nuclei']:,} nuclei "
            f"(prevalence {v['prevalence']:.4f}); predictions {v['source']}."
        ),
        "",
        "## (a) Prevalence bins (Youden threshold)",
        "",
        (
            f"Median [IQR] per bin. Calibrated AUPRC (π₀ = {q0}) over slides with ≥ {config.min_carcinoma_nuclei} "
            f"carcinoma nuclei only ({t['positive_slides_unreliable_AUPRC']} of {t['positive_slides']} positive slides "
            "have fewer: their AUPRC values are unreliable; recall and FPR include them)."
        ),
        "",
        "| Group | Bin | Slides | Cal. AUPRC | Recall | FPR |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    def m(r: pd.Series, col: str, f: str = ".3f") -> str:
        if np.isnan(r[f"{col}_median"]):
            return "–"
        return f"{r[f'{col}_median']:{f}} [{r[f'{col}_q1']:{f}}, {r[f'{col}_q3']:{f}}]"

    for _, r in bins.iterrows():
        lines.append(
            f"| {r.group} | {r.prevalence_bin} | {r.n_slides} ({r.n_slides_reliable_AUPRC}) | {m(r, cal)} | "
            f"{m(r, 'recall_youden')} | {m(r, 'FPR_youden', '.4f')} |"
        )
    lines += [
        "",
        "Slides: all (with reliable AUPRC).",
        "",
        "## (b) Spearman correlation with slide prevalence (95% bootstrap CI over slides)",
        "",
        f"| Group | Recall (Youden) | Recall (0.90) | Cal. AUPRC (π₀ = {q0}) | Cal. AUPRC (π₀ = {config.pi0[1]}) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for g in GROUPS:
        lines.append(
            f"| {g} | {fmt_rho(corr, g, 'recall_youden')} | {fmt_rho(corr, g, 'recall_recall90')} | "
            f"{fmt_rho(corr, g, cal)} | {fmt_rho(corr, g, f'calibrated_AUPRC_pi0_{config.pi0[1]}')} |"
        )
    qa = quant[quant["group"] == "all"]
    lines += [
        "",
        "## (c) Seed-mean scores per prevalence bin (all positive slides)",
        "",
        "| Bin | Class | Slides | Nuclei | p5 | p25 | p50 | p75 | p95 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in [
        *qa.itertuples(),
        *quant[quant["group"] == "negative slides"].itertuples(),
    ]:
        lines.append(
            f"| {r.prevalence_bin} | {r[3]} | {r.n_slides} | {r.n:,} | {r.p5:.3f} | {r.p25:.3f} | "
            f"{r.p50:.3f} | {r.p75:.3f} | {r.p95:.3f} |"
        )
    lines += [
        "",
        (
            f"Negative slides ({len(neg)}): FPR median {neg['FPR_youden'].median():.4f} at the Youden threshold "
            f"(95th percentile {neg['FPR_youden'].quantile(0.95):.4f}), {neg['FPR_recall90'].median():.4f} at the "
            f"recall-0.90 threshold."
        ),
        "",
        "## Verdict",
        "",
        verdict(corr, cal, pos),
        "",
        "## Checks",
        "",
        f"- Pooled per-seed AUPRC equals the results run's ({', '.join(f'{a:.4f}' for a in t['pooled_AUPRC_per_seed'])}); same nuclei and labels.",
        f"- Calibrated AUPRC with c = 1 equals the torchmetrics AUPRC per slide and seed (max |Δ| {checks['calibration_check']['max_abs_diff_vs_torchmetrics']:.1e}).",
        f"- ISUP groups of the positive slides: {t['positive_slides_by_isup_group']} ({t['positive_slides_isup_0']} with ISUP 0 in PANDA: in 'all' only).",
    ]
    rr = checks["rerun_check"]
    if rr:
        lines.append(
            "- Re-prediction of the test set at the evaluation commit vs the stored predictions: mean |Δp| "
            + ", ".join(f"{s['mean']:.4f}" for s in rr["per_seed_abs_diff"])
            + "; 99th percentile "
            + ", ".join(f"{s['q99']:.3f}" for s in rr["per_seed_abs_diff"])
            + "; max "
            + ", ".join(f"{s['max']:.3f}" for s in rr["per_seed_abs_diff"])
            + "; pooled AUPRC "
            + ", ".join(f"{a:.4f}" for a in rr["pooled_AUPRC_rerun"])
            + " vs "
            + ", ".join(f"{a:.4f}" for a in rr["pooled_AUPRC_stored"])
            + "."
        )
    lines.append("")
    return "\n".join(lines)


def verdict(corr: pd.DataFrame, cal: str, pos: pd.DataFrame) -> str:
    rec = {g: ci_sign(corr, g, "recall_youden") for g in GROUPS}
    auc = {g: ci_sign(corr, g, cal) for g in GROUPS}
    word = {1: "increases", -1: "decreases", 0: "shows no clear trend"}
    parts = [
        (
            f"Recall at the Youden threshold {word[rec['all']]} with slide prevalence overall "
            f"({fmt_rho(corr, 'all', 'recall_youden')}); calibrated AUPRC {word[auc['all']]} "
            f"({fmt_rho(corr, 'all', cal)})."
        )
    ]
    within = [g for g in GROUPS[1:] if rec[g] == 1]
    if rec["all"] == 1:
        shift = (
            "The MMCI pattern (carcinoma scores shifted below the threshold where there is little carcinoma) "
            "replicates on Radboud"
        )
        r_rec, r_auc = rho(corr, "all", "recall_youden"), rho(corr, "all", cal)
        if auc["all"] == 1 and r_auc < r_rec / 2:
            shift += (
                "; the within-slide ranking degrades only weakly with lower prevalence, so it is mainly a shift "
                "of the carcinoma scores, as on MMCI"
            )
        elif auc["all"] == 1:
            shift += ", but the within-slide ranking degrades about as strongly, so it is not only a score shift"
        else:
            shift += "; the within-slide ranking does not degrade, so it is a shift of the carcinoma scores, as on MMCI"
        parts.append(shift + ".")
        if ci_sign(corr, "all", "FPR_youden") == 1:
            parts.append(
                "Benign nuclei shift the same way: the FPR rises with prevalence "
                f"({fmt_rho(corr, 'all', 'FPR_youden')}), i.e. the scores of the whole slide follow the amount of "
                "carcinoma in it."
            )
        if len(within) == 2:
            parts.append(
                "It holds within ISUP 1 and within ISUP ≥ 2, so it is not explained by grade."
            )
        elif within:
            parts.append(
                f"Within grade groups it holds only for {within[0]} ("
                + "; ".join(
                    f"{g}: {fmt_rho(corr, g, 'recall_youden')}" for g in GROUPS[1:]
                )
                + ")."
            )
        else:
            parts.append(
                "Within grade groups it does not hold ("
                + "; ".join(
                    f"{g}: {fmt_rho(corr, g, 'recall_youden')}" for g in GROUPS[1:]
                )
                + "), so on Radboud it is largely explained by grade."
            )
    else:
        parts.append(
            "The MMCI pattern does not replicate on Radboud at the slide level."
        )
    return " ".join(parts)


def preflight(out: Path, allowed: set[str]) -> None:
    """No credentials; every 32-hex token is a public PANDA image id or one of the runs."""
    for path in out.iterdir():
        if path.suffix not in {".csv", ".json", ".md", ".yaml"}:
            continue
        text = path.read_text()
        assert not CREDENTIAL_URL.search(text), f"credential URL in {path.name}"
        unknown = set(HEX32.findall(text)) - allowed
        assert not unknown, f"{len(unknown)} non-public ids in {path.name}"


@with_cli_args(["+postprocessing=slide_error_analysis_radboud"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
def main(config: DictConfig) -> None:
    # download_artifacts resolves mlflow-artifacts:/ against the environment variable
    os.environ["MLFLOW_TRACKING_URI"] = str(config.mlflow_tracking_uri)
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    client = MlflowClient()
    out, checks, allowed = compute(config, client)
    (out / "config-resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    preflight(out, allowed)
    if not config.log_mlflow:
        print(f"log_mlflow=false: outputs in {out}")
        return

    th = checks["thresholds"]
    params = {
        "results_run": config.results_run,
        "dataset": config.dataset_label,
        "pi0": ",".join(str(v) for v in config.pi0),
        "threshold_youden": th["youden"]["threshold"],
        "threshold_recall_0.90": th["recall90"]["threshold"],
        "min_carcinoma_nuclei": config.min_carcinoma_nuclei,
        "evaluation_commit": checks["evaluation_commit"],
        "validation_predictions": checks["validation"]["source"],
        **{f"eval_run_{i}": r for i, r in enumerate(config.eval_runs, start=1)},
        **{f"training_run_{i}": r for i, r in enumerate(config.training_runs, start=1)},
    }
    tags = {
        "mlflow.runName": config.metadata.run_name,
        "mlflow.note.content": config.metadata.description,
        "results_run": config.results_run,
        **git_info(*CODE),
    }
    for text in [*map(str, tags.values()), *map(str, params.values())]:
        assert not CREDENTIAL_URL.search(text), "credential URL in a tag or param"
    run_id = client.create_run(
        str(config.metadata.experiment_id), run_name=config.metadata.run_name, tags=tags
    ).info.run_id
    user = client.get_run(config.results_run).data.tags.get("mlflow.user")
    if user:
        client.set_tag(run_id, "mlflow.user", user)
    for k, v in params.items():
        client.log_param(run_id, k, v)
    for name in ("youden", "recall90"):
        client.log_metric(run_id, f"threshold/{name}", th[name]["threshold"])
        client.log_metric(
            run_id, f"validation/recall_{name}", th[name]["validation"]["recall"]
        )
        client.log_metric(
            run_id, f"validation/FPR_{name}", th[name]["validation"]["FPR"]
        )
    client.log_metric(
        run_id,
        "test/positive_slides_unreliable_AUPRC",
        checks["test"]["positive_slides_unreliable_AUPRC"],
    )
    for name in (
        "per_slide_radboud.csv",
        "prevalence_bins_radboud.csv",
        "spearman_radboud.csv",
        "score_quantiles_radboud.csv",
        "thresholds.json",
        "report.md",
    ):
        client.log_artifact(run_id, str(out / name))
    client.log_artifact(run_id, str(out / "config-resolved.yaml"), "code")
    for source in CODE:
        client.log_artifact(run_id, str(REPO / source), "code")

    r = client.get_run(run_id)
    leaked = [
        k
        for k, v in {**r.data.tags, **r.data.params}.items()
        if CREDENTIAL_URL.search(v)
    ]
    assert not leaked, f"credential URL in run tags/params {leaked}"
    client.set_terminated(run_id)
    print(
        f"MLflow run {run_id} (experiment {config.metadata.experiment_id}); outputs in {out}"
    )


if __name__ == "__main__":
    main()
