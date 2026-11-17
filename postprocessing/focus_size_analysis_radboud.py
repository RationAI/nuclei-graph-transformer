# ruff: noqa: RUF001, RUF002, RUF003  (Greek letters / minus signs in report text are intended)
r"""Why do nucleus-level scores fall with the amount of carcinoma? Focus size, boundary, pattern.

PANDA Radboud test set (same slides, labels and QC as tab:res-nucleus, run 6aa01c04): labels
``annot_label`` (a nucleus is carcinoma when ≥ 50 % of its polygon vertices lie in the G3–5 mask),
missing → 0, nuclei of ``is_carcinoma = False`` slides → 0. Six models (``models``), three seeds
each, stored test predictions of their evaluation runs; score = seed-mean probability. Existing
checkpoints only, no training.

Stages (``stages``):

``validation``
    Each model's Youden threshold (max TPR − FPR over the distinct scores, highest t on ties) on the
    seed-mean probability of the validation nuclei (the training runs' fixed split of the training
    slides, ``split_train_val``, random_state = 42; same label rule). Validation predictions are made
    with each evaluation run's own commit (git worktree, ``nucleus_inference_at_commit.py``) and the
    training run's checkpoint used by that evaluation run; the reproduction is checked by also
    re-predicting ``inference.test_sample_slides`` test slides and comparing with the stored ones.
``geometry``
    Per test slide, from the Radboud label mask at level 0 (0 background, 1 stroma, 2 benign
    epithelium, 3–5 Gleason pattern): per nucleus, the mask values under its polygon vertices
    (the vertex rule of the thesis labels, which must reproduce ``annot_label``) give the class:
    carcinoma nuclei the Gleason pattern with most vertices (ties → higher), the others benign
    epithelium / stroma when ≥ 50 % of the vertices are 2 / 1, else other. The carcinoma mask
    (pattern ≥ 3) subsampled by ``geometry.stride`` is closed with a disk of each diameter in
    ``geometry.closings_um`` (0 = none); foci = 8-connected components of the closed mask. Radboud
    masks are epithelium-level (stroma between carcinoma glands is 1), so without a closing a focus
    is a gland. Per nucleus and closing: the focus of its centroid (nearest focus when outside) and
    the signed distance (µm; + inside, − outside) from its centroid to the boundary of the closed
    mask. µm per pixel: the slide table's ``mpp_x`` × stride.
``analysis``
    D1 per focus (carcinoma nuclei assigned by centroid): area (closed region and carcinoma
    epithelium, mm²), carcinoma nuclei, dominant pattern, median score and recall at the threshold;
    Spearman ρ of area with median score / recall (foci with ≥ ``min_focus_nuclei`` carcinoma
    nuclei), percentile CI from a bootstrap over slides. D2 signed-distance profile (10 µm bins,
    −200…+200 µm): median score, recall (carcinoma nuclei) and FPR (non-carcinoma nuclei) per bin,
    all foci and foci < ``small_focus_mm2`` / > ``large_focus_mm2``. D3 recall per pattern,
    within-slide AUROC of each pattern vs the slide's non-carcinoma nuclei, D1 within G3- (and G4-)
    dominant foci, and the partial Spearman of per-slide recall vs carcinoma share given the slide's
    G3 fraction.

Outputs in ``out_dir`` (tables without slide ids; slides are referred to by their row in the sorted
test table): CSVs, PDFs and ``report.md``; logged to a new MLflow run in experiment 37.
"""

import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
from collections import defaultdict
from itertools import pairwise
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
from postprocessing.focus_geometry_radboud import slide_geometry
from postprocessing.nucleus_results_mmci import CREDENTIAL_URL
from postprocessing.patient_error_analysis_mmci import operating_point
from postprocessing.patient_score_quantiles_mmci import youden_threshold
from postprocessing.tile.bootstrap_pixel_vs_nuclei import git_info


REPO = Path(__file__).parents[1]
INFERENCE = "postprocessing/nucleus_inference_at_commit.py"
CODE = (
    "postprocessing/focus_size_analysis_radboud.py",
    "postprocessing/focus_geometry_radboud.py",
    "postprocessing/nucleus_inference_at_commit.py",
    "visualization/focus_size_figures.py",
    "configs/postprocessing/focus_size_analysis_radboud.yaml",
)
CKPT_EPOCH = re.compile(r"epoch=(\d+)")
HEX32 = re.compile(r"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])")
COMMIT_TAGS = ("mlflow.source.git.commit", "code_commit", "git.commit", "git_commit")


# --------------------------------------------------------------------------
# Runs and data
# --------------------------------------------------------------------------


def remap(value, config: DictConfig):
    """Paths of the public cluster's project folder, as mounted here (recursively in dicts / lists)."""
    remote, local = str(config.remote_project_path), str(config.local_project_path)
    if isinstance(value, str):
        return local + value[len(remote) :] if value.startswith(remote) else value
    if isinstance(value, dict):
        return {k: remap(v, config) for k, v in value.items()}
    if isinstance(value, list):
        return [remap(v, config) for v in value]
    return value


def all_models(config: DictConfig) -> dict[str, DictConfig]:
    return {**config.models, **config.sensitivity_models}


def run_info(client: MlflowClient, eval_run: str, tmp: Path) -> dict:
    """Training run, commit, checkpoint artifact and resolved configs of an evaluation run."""
    tags = client.get_run(eval_run).data.tags
    training = tags["mlflow.parentRunId"]
    commit = next((tags[k] for k in COMMIT_TAGS if k in tags), None)
    assert commit, f"{eval_run}: no commit tag"
    (tmp / eval_run).mkdir(parents=True, exist_ok=True)
    (tmp / training).mkdir(parents=True, exist_ok=True)
    eval_cfg = yaml.safe_load(
        Path(
            client.download_artifacts(
                eval_run, "configs/config-resolved.yaml", str(tmp / eval_run)
            )
        ).read_text()
    )
    train_cfg = yaml.safe_load(
        Path(
            client.download_artifacts(
                training, "configs/config-resolved.yaml", str(tmp / training)
            )
        ).read_text()
    )
    files = [f.path for f in client.list_artifacts(training, "checkpoints")]
    assert len(files) == 1, f"{training}: {len(files)} checkpoints"
    epoch = int(CKPT_EPOCH.search(files[0]).group(1))
    used = eval_cfg.get("checkpoint")
    if used:  # the evaluation run's checkpoint must be the training run's only one
        assert training in used and int(CKPT_EPOCH.search(used).group(1)) == epoch, (
            f"{eval_run}: evaluated {used}, training run holds {files[0]}"
        )
    return {
        "eval_run": eval_run,
        "training_run": training,
        "commit": commit.split()[0],
        "checkpoint": files[0],
        "checkpoint_epoch": epoch,
        "checkpoint_source": "evaluation config"
        if used
        else "training run's only checkpoint",
        "train_cfg": train_cfg,
    }


def test_slides(config: DictConfig) -> pd.DataFrame:
    meta = pd.read_parquet(download_artifacts(config.metadata_uri)).sort_values(
        "slide_id", ignore_index=True
    )
    meta["slide_nuclei_path"] = [remap(p, config) for p in meta["slide_nuclei_path"]]
    return meta


def validation_slides(config: DictConfig, cfgs: list[dict]) -> pd.DataFrame:
    keys = ("metadata", "split_size", "split_stratify_col", "split_group_col")
    for c in cfgs:
        assert {k: c["data"][k] for k in keys} == {
            k: cfgs[0]["data"][k] for k in keys
        }, "training runs with different validation splits"
    d0 = cfgs[0]["data"]
    train_meta = pd.read_parquet(download_artifacts(d0["metadata"]["train"]))
    _, val = split_train_val(
        train_meta, d0["split_size"], d0["split_stratify_col"], d0["split_group_col"]
    )
    val = val.sort_values("slide_id", ignore_index=True)
    val["slide_nuclei_path"] = [remap(p, config) for p in val["slide_nuclei_path"]]
    return val


def labelled(
    config: DictConfig, slides: pd.DataFrame, frames: list[pd.DataFrame]
) -> tuple[pd.DataFrame, NDArray, NDArray, int]:
    """(slide, key) of every nucleus, labels, probabilities per seed, unmatched count."""
    ref = frames[0][["slide", "key"]]
    for f in frames[1:]:
        assert np.array_equal(f["slide"].to_numpy(), ref["slide"].to_numpy())
        assert np.array_equal(f["key"].to_numpy(), ref["key"].to_numpy()), (
            "seeds differ"
        )
    carcinoma = slides["is_carcinoma"].to_numpy(bool)
    labels = load_labels(slides, Path(config.supervision_dir), config.label_column)
    y, unmatched = label_predictions(ref, labels, carcinoma)
    probs = np.stack([f["p"].to_numpy(np.float32) for f in frames])
    return ref.reset_index(drop=True), y, probs, unmatched


def stored_test(
    config: DictConfig, eval_run: str, slide_ids: list[str]
) -> pd.DataFrame:
    dirs = [Path(d) for d in config.test_cache_dirs]
    for d in dirs:
        if (d / f"{eval_run}.parquet").exists() and (d / f"{eval_run}.json").exists():
            return load_run(eval_run, slide_ids, d, config.predictions_artifact)[0]
    df, info = load_run(eval_run, slide_ids, dirs[-1], config.predictions_artifact)
    assert info["test_slides_without_predictions"] == 0, f"{eval_run}: {info}"
    return df


# --------------------------------------------------------------------------
# Inference at the evaluation commit
# --------------------------------------------------------------------------


def predict_at_commit(
    client: MlflowClient,
    config: DictConfig,
    commit: str,
    infos: list[dict],
    slides: pd.DataFrame,
    outputs: list[Path],
    tmp: Path,
) -> list[pd.DataFrame]:
    """Predicts `slides` with each run's checkpoint and the model code of `commit`."""
    if all(p.exists() for p in outputs):
        return [pd.read_parquet(p) for p in outputs]
    runs = []
    for info in infos:
        t = info["training_run"]
        local = Path(client.download_artifacts(t, info["checkpoint"], str(tmp / t)))
        if local.is_dir():
            (local,) = list(local.rglob("*.ckpt"))
        cfg_path = tmp / t / "config-remapped.yaml"
        cfg_path.write_text(yaml.safe_dump(remap(info["train_cfg"], config)))
        runs.append({"checkpoint": str(local), "config": str(cfg_path)})
    stem = outputs[0].stem.split("_")[-1]
    slides_path = tmp / f"{commit[:8]}_{stem}_slides.parquet"
    slides.to_parquet(slides_path, index=False)
    job = tmp / f"{commit[:8]}_{stem}_{outputs[0].stem[:8]}_job.json"
    job.write_text(
        json.dumps(
            {
                "slides": str(slides_path),
                "runs": runs,
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


def model_thresholds(
    client: MlflowClient,
    config: DictConfig,
    name: str,
    infos: list[dict],
    val: pd.DataFrame,
    meta: pd.DataFrame,
    tmp: Path,
) -> dict:
    """Youden threshold of the seed-mean validation probability and the reproduction check."""
    cache = Path(config.inference_cache_dir)
    by_commit: dict[str, list[int]] = defaultdict(list)
    for i, info in enumerate(infos):
        by_commit[info["commit"]].append(i)

    rng = np.random.default_rng(int(config.inference.test_sample_seed))
    sample_idx = np.sort(
        rng.choice(len(meta), int(config.inference.test_sample_slides), replace=False)
    )
    sample = meta.iloc[sample_idx].reset_index(drop=True)

    val_frames: list[pd.DataFrame | None] = [None] * len(infos)
    sample_frames: list[pd.DataFrame | None] = [None] * len(infos)
    for commit, idx in by_commit.items():
        group = [infos[i] for i in idx]
        outs_v = [
            cache / commit[:8] / f"{g['training_run']}_validation.parquet"
            for g in group
        ]
        outs_t = [
            cache / commit[:8] / f"{g['training_run']}_testsample.parquet"
            for g in group
        ]
        for i, f in zip(
            idx,
            predict_at_commit(client, config, commit, group, val, outs_v, tmp),
            strict=True,
        ):
            val_frames[i] = f
        for i, f in zip(
            idx,
            predict_at_commit(client, config, commit, group, sample, outs_t, tmp),
            strict=True,
        ):
            sample_frames[i] = f

    _, vy, vprobs, val_unmatched = labelled(config, val, val_frames)
    vmean = vprobs.astype(np.float64).mean(0)
    t, j = youden_threshold(vy, vmean)

    # reproduction: re-predicted test sample vs the stored test predictions of the same nuclei
    rerun = []
    for info, f in zip(infos, sample_frames, strict=True):
        stored = stored_test(config, info["eval_run"], meta["slide_id"].tolist())
        stored = stored[stored["slide"].isin(sample_idx)].copy()
        stored["slide"] = np.searchsorted(sample_idx, stored["slide"].to_numpy())
        m = stored.merge(f, on=["slide", "key"], suffixes=("_stored", "_rerun"))
        assert len(m) == len(stored) == len(f), (
            f"{info['eval_run']}: nuclei of the re-predicted sample differ"
        )
        d = np.abs(m["p_stored"].to_numpy() - m["p_rerun"].to_numpy())
        rerun.append(
            {
                "eval_run": info["eval_run"],
                "nuclei": len(d),
                "mean_abs_diff": float(d.mean()),
                "q99_abs_diff": float(np.quantile(d, 0.99)),
                "max_abs_diff": float(d.max()),
            }
        )
    return {
        "model": name,
        "threshold": t,
        "J": j,
        "validation": operating_point(vy, vmean, t),
        "validation_nuclei": len(vy),
        "validation_carcinoma_nuclei": int(vy.sum()),
        "validation_slides": len(val),
        "validation_unmatched": val_unmatched,
        "rerun_test_sample": rerun,
        "runs": [{k: v for k, v in info.items() if k != "train_cfg"} for info in infos],
    }


def stage_validation(config: DictConfig, client: MlflowClient) -> dict:
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = Path(config.tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    meta = test_slides(config)
    path = out / "thresholds.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    only = config.get("validation_models")
    for name, m in all_models(config).items():
        if name in done or (only and name not in only):
            continue
        infos = [run_info(client, e, tmp) for e in m.eval_runs]
        val = validation_slides(config, [i["train_cfg"] for i in infos])
        assert not set(val["slide_id"]) & set(meta["slide_id"]), "validation in test"
        res = model_thresholds(client, config, name, infos, val, meta, tmp)
        if m.get("thesis_threshold") is not None:
            res["thesis_threshold"] = float(m.thesis_threshold)
            assert abs(res["threshold"] - float(m.thesis_threshold)) < 0.005, (
                f"{name}: Youden {res['threshold']:.4f} != thesis {m.thesis_threshold}"
            )
        done[name] = res
        path.write_text(json.dumps(done, indent=1, default=str))
        print(
            f"{name}: Youden {res['threshold']:.4f} (J {res['J']:.4f}); rerun max |Δp| "
            + ", ".join(f"{r['max_abs_diff']:.3g}" for r in res["rerun_test_sample"])
        )
    return done


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def stage_geometry(config: DictConfig) -> pd.DataFrame:
    meta = test_slides(config)
    g = config.geometry
    jobs = []
    for s, row in enumerate(meta.itertuples()):
        jobs.append(
            {
                "slide": s,
                "slide_id": row.slide_id,
                "out_dir": str(config.geometry_dir),
                "nuclei_path": row.slide_nuclei_path,
                "mask_path": str(
                    Path(config.label_masks_dir) / f"{row.slide_id}_mask.tiff"
                ),
                "slide_path": row.slide_path,
                "label_path": str(
                    Path(config.supervision_dir) / f"{row.slide_id}.parquet"
                )
                if row.is_carcinoma
                else None,
                "label_column": config.label_column,
                "overlap_threshold": float(config.overlap_threshold),
                "stride": int(g.stride),
                "mpp": float(row.mpp_x),
                "closings_um": [int(d) for d in g.closings_um],
            }
        )
    missing = [j["slide_id"] for j in jobs if not Path(j["mask_path"]).exists()]
    assert not missing, f"{len(missing)} test slides without a label mask"
    with mp.get_context("spawn").Pool(int(g.workers), maxtasksperchild=20) as pool:
        checks = list(
            tqdm(
                pool.imap_unordered(slide_geometry, jobs),
                total=len(jobs),
                desc="geometry",
            )
        )
    checks = pd.DataFrame(checks).sort_values("slide", ignore_index=True)
    bad = checks["label_mismatch"].fillna(0).astype(int)
    assert (bad == 0).all(), (
        f"vertex rule != {config.label_column} on {int((bad > 0).sum())} slides"
    )
    return checks


def load_geometry(
    config: DictConfig, meta: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gdir = Path(config.geometry_dir)
    nuclei = pd.concat(
        [pd.read_parquet(gdir / "nuclei" / f"{s}.parquet") for s in meta["slide_id"]],
        ignore_index=True,
    ).sort_values(["slide", "key"], ignore_index=True)
    foci = [pd.read_parquet(gdir / "foci" / f"{s}.parquet") for s in meta["slide_id"]]
    foci = pd.concat([f for f in foci if len(f)], ignore_index=True)
    return nuclei, foci


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def spearman(x: NDArray, y: NDArray) -> float:
    if len(x) < 3:
        return np.nan
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def partial_spearman(x: NDArray, y: NDArray, z: NDArray) -> float:
    """Spearman correlation of x and y given z (partial correlation of the ranks)."""
    r = np.corrcoef(np.stack([rankdata(x), rankdata(y), rankdata(z)]))
    rxy, rxz, ryz = r[0, 1], r[0, 2], r[1, 2]
    return float((rxy - rxz * ryz) / np.sqrt((1 - rxz**2) * (1 - ryz**2)))


def auroc(pos: NDArray, neg: NDArray) -> float:
    r = rankdata(np.concatenate([pos, neg]))
    n1, n0 = len(pos), len(neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


class SlideBootstrap:
    """Resamples of the positive slides (with replacement), shared by every statistic.

    ``draws`` hold slide indices (rows of the test table); ``counts`` how often each slide is drawn.
    """

    def __init__(self, config: DictConfig, n_slides: int, slides: NDArray) -> None:
        b = config.bootstrap
        rng = np.random.default_rng(int(b.seed))
        draws = slides[rng.integers(0, len(slides), (int(b.n_resamples), len(slides)))]
        self.draws = draws
        self.counts = np.stack([np.bincount(d, minlength=n_slides) for d in draws])
        self.alpha = (1 - float(b.ci_level)) / 2 * 100

    def ci(self, reps: NDArray) -> tuple[float, float]:
        reps = np.asarray(reps, float)
        reps = reps[np.isfinite(reps)]
        if len(reps) == 0:
            return np.nan, np.nan
        return (
            float(np.percentile(reps, self.alpha)),
            float(np.percentile(reps, 100 - self.alpha)),
        )

    def ratio(self, num: NDArray, den: NDArray) -> tuple[float, float]:
        """CI of sum(num) / sum(den) for per-slide sums (slides on axis 0)."""
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.ci((self.counts @ num) / (self.counts @ den))

    def grouped(
        self, slide_of_item: NDArray, stat, *arrays: NDArray
    ) -> tuple[float, float]:
        """CI of stat(*arrays[items of the resampled slides]) (items grouped by slide)."""
        order = np.argsort(slide_of_item, kind="stable")
        bounds = np.searchsorted(
            slide_of_item[order], np.arange(self.counts.shape[1] + 1)
        )
        ranges = [order[bounds[s] : bounds[s + 1]] for s in range(len(bounds) - 1)]
        reps = []
        for d in self.draws:
            idx = np.concatenate([ranges[s] for s in d])
            reps.append(stat(*(a[idx] for a in arrays)) if len(idx) else np.nan)
        return self.ci(reps)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


PATTERNS = (3, 4, 5)
CLS_NAMES = {0: "other", 1: "stroma", 2: "benign epithelium", 3: "G3", 4: "G4", 5: "G5"}


def assemble(config: DictConfig, thresholds: dict) -> dict:
    """Nucleus table (geometry + labels + seed-mean score per model), foci and slides."""
    meta = test_slides(config)
    nuclei, foci = load_geometry(config, meta)
    scores, seeds = {}, {}
    ref = None
    for name, m in all_models(config).items():
        frames = [
            stored_test(config, e, meta["slide_id"].tolist()) for e in m.eval_runs
        ]
        keys, y, probs, unmatched = labelled(config, meta, frames)
        assert unmatched == 0, (
            f"{name}: {unmatched} carcinoma-slide nuclei without labels"
        )
        if ref is None:
            ref = keys
            assert np.array_equal(keys["slide"].to_numpy(), nuclei["slide"].to_numpy())
            assert np.array_equal(keys["key"].to_numpy(), nuclei["key"].to_numpy()), (
                "prediction nuclei != geometry nuclei"
            )
            nuclei["y"] = y
        else:
            assert np.array_equal(keys["key"].to_numpy(), ref["key"].to_numpy())
            assert np.array_equal(y, nuclei["y"].to_numpy())
        scores[name] = probs.astype(np.float64).mean(0).astype(np.float32)
        seeds[name] = len(probs)
        del probs, frames
    y = nuclei["y"].to_numpy()
    carcinoma_slide = meta["is_carcinoma"].to_numpy(bool)[nuclei["slide"].to_numpy()]
    rule_vs_label = {
        "carcinoma_slides_mismatch": int(
            ((nuclei["y_rule"].to_numpy() != y) & carcinoma_slide).sum()
        ),
        "negative_slides_rule_carcinoma": int(
            ((nuclei["y_rule"].to_numpy() == 1) & ~carcinoma_slide).sum()
        ),
    }
    assert rule_vs_label["carcinoma_slides_mismatch"] == 0
    # class of nuclei labelled 0 by the slide rule but carcinoma by the mask: other
    cls = nuclei["cls"].to_numpy().copy()
    cls[(y == 0) & (cls >= 3)] = 0
    nuclei["cls"] = cls
    per_slide = nuclei.groupby("slide").agg(
        n=("y", "size"),
        n_carcinoma=("y", "sum"),
        n_g3=("cls", lambda c: int((c == 3).sum())),
        n_g4=("cls", lambda c: int((c == 4).sum())),
        n_g5=("cls", lambda c: int((c == 5).sum())),
    )
    pos_slides = np.flatnonzero(per_slide["n_carcinoma"].to_numpy() > 0)
    return {
        "meta": meta,
        "nuclei": nuclei,
        "foci": foci,
        "scores": scores,
        "seeds": seeds,
        "thresholds": {k: float(v["threshold"]) for k, v in thresholds.items()},
        "per_slide": per_slide,
        "positive_slides": pos_slides,
        "rule_vs_label": rule_vs_label,
    }


def focus_table(data: dict, closing: int) -> pd.DataFrame:
    """Per focus (carcinoma nuclei by centroid): areas, nuclei per pattern, dominant pattern."""
    n = data["nuclei"]
    car = n["y"].to_numpy() == 1
    f = pd.DataFrame(
        {
            "slide": n["slide"].to_numpy()[car],
            "focus": n[f"focus_{closing}"].to_numpy()[car],
            "cls": n["cls"].to_numpy()[car],
        }
    )
    assert (f["focus"] >= 1).all(), "carcinoma nucleus without a focus"
    t = (
        f.assign(**{f"n_g{p}": (f["cls"] == p).astype(int) for p in PATTERNS})
        .groupby(["slide", "focus"], as_index=False)
        .agg(
            n_carcinoma=("cls", "size"),
            **{f"n_g{p}": (f"n_g{p}", "sum") for p in PATTERNS},
        )
    )
    counts = t[[f"n_g{p}" for p in (5, 4, 3)]].to_numpy()
    t["dominant"] = 5 - np.argmax(counts, axis=1)  # ties -> higher pattern
    t["dominant_share"] = counts.max(1) / t["n_carcinoma"]
    areas = data["foci"][data["foci"]["closing_um"] == closing]
    t = t.merge(
        areas[
            [
                "slide",
                "focus",
                "area_mm2",
                "carcinoma_area_mm2",
                "g3_area_mm2",
                "g4_area_mm2",
                "g5_area_mm2",
            ]
        ],
        on=["slide", "focus"],
        how="left",
        validate="1:1",
    )
    assert t["area_mm2"].notna().all()
    t["focus_index"] = np.arange(len(t))
    return t


def focus_scores(
    data: dict, closing: int, ft: pd.DataFrame, model: str
) -> pd.DataFrame:
    n = data["nuclei"]
    car = n["y"].to_numpy() == 1
    s = data["scores"][model][car]
    t = data["thresholds"][model]
    d = pd.DataFrame(
        {
            "slide": n["slide"].to_numpy()[car],
            "focus": n[f"focus_{closing}"].to_numpy()[car],
            "score": s,
            "hit": (s >= t).astype(np.int64),
        }
    )
    g = d.groupby(["slide", "focus"], as_index=False).agg(
        median_score=("score", "median"), tp=("hit", "sum")
    )
    out = ft.merge(g, on=["slide", "focus"], how="left", validate="1:1")
    out["recall"] = out["tp"] / out["n_carcinoma"]
    return out


def d1(
    config: DictConfig, data: dict, boot: SlideBootstrap
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    a = config.analysis
    bins = [float(v) for v in a.area_bins_mm2]
    labels = [f"{lo:g}-{hi:g}" for lo, hi in pairwise(bins)]
    corr_rows, bin_rows, per_focus = [], [], {}
    for closing in config.geometry.closings_um:
        closing = int(closing)
        ft = focus_table(data, closing)
        for model in all_models(config):
            fs = focus_scores(data, closing, ft, model)
            fs.insert(0, "model", model)
            fs.insert(1, "closing_um", closing)
            per_focus[(model, closing)] = fs
            ok = fs[fs["n_carcinoma"] >= int(a.min_focus_nuclei)]
            subsets = {"all": ok}
            if closing == int(config.geometry.primary_closing_um):
                for p in PATTERNS:
                    subsets[f"G{p}-dominant"] = ok[ok["dominant"] == p]
                    subsets[f"G{p}-pure"] = ok[
                        (ok["dominant"] == p)
                        & (ok["dominant_share"] >= float(a.pure_share))
                    ]
            for subset, d in subsets.items():
                for var in ("median_score", "recall"):
                    x, v = d["area_mm2"].to_numpy(), d[var].to_numpy()
                    lo, hi = boot.grouped(d["slide"].to_numpy(), spearman, x, v)
                    corr_rows.append(
                        {
                            "model": model,
                            "closing_um": closing,
                            "foci": subset,
                            "variable": var,
                            "n_foci": len(d),
                            "n_slides": d["slide"].nunique(),
                            "spearman_rho": spearman(x, v),
                            "ci_lo": lo,
                            "ci_hi": hi,
                        }
                    )
            # area bins: median of the focus medians, pooled recall
            cut = pd.cut(ok["area_mm2"], bins=bins, labels=labels, right=False)
            for subset_name, sub in (
                ("all", ok),
                *[(f"G{p}-dominant", ok[ok["dominant"] == p]) for p in PATTERNS],
            ):
                c = cut.loc[sub.index]
                for lab in labels:
                    d = sub[c == lab]
                    if d.empty:
                        continue
                    q1, med, q3 = np.quantile(d["median_score"], [0.25, 0.5, 0.75])
                    bin_rows.append(
                        {
                            "model": model,
                            "closing_um": closing,
                            "foci": subset_name,
                            "area_bin_mm2": lab,
                            "n_foci": len(d),
                            "n_slides": d["slide"].nunique(),
                            "carcinoma_nuclei": int(d["n_carcinoma"].sum()),
                            "median_area_mm2": float(d["area_mm2"].median()),
                            "median_score_median": float(med),
                            "median_score_q1": float(q1),
                            "median_score_q3": float(q3),
                            "recall_pooled": float(
                                d["tp"].sum() / d["n_carcinoma"].sum()
                            ),
                            "recall_focus_median": float(d["recall"].median()),
                            "share_g3": float(d["n_g3"].sum() / d["n_carcinoma"].sum()),
                        }
                    )
    return pd.DataFrame(corr_rows), pd.DataFrame(bin_rows), per_focus


def d2(
    config: DictConfig, data: dict, boot: SlideBootstrap, size_class: NDArray
) -> pd.DataFrame:
    """Signed-distance profiles; boundaries of the primary closing (foci) and of no closing (glands).

    `size_class` per nucleus: 1 = its (nearest) focus at the primary closing is small, 2 = large,
    0 = neither. Positive slides only.
    """
    e = config.analysis.distance_edges_um
    edges = np.arange(float(e.lo), float(e.hi) + 1e-9, float(e.step))
    nb = len(edges) + 1  # 0 = below lo, len(edges) = above hi
    n = data["nuclei"]
    n_slides = len(data["meta"])
    pos = np.zeros(n_slides, bool)
    pos[data["positive_slides"]] = True
    slide_all = n["slide"].to_numpy().astype(np.int64)
    y_all = n["y"].to_numpy()
    primary = int(config.geometry.primary_closing_um)
    rows = []
    for boundary in (primary, 0):
        sd_all = n[f"sd_{boundary}"].to_numpy()
        base = pos[slide_all] & np.isfinite(sd_all)
        for restriction, code in (("all", None), ("small", 1), ("large", 2)):
            idx = np.flatnonzero(base if code is None else base & (size_class == code))
            slide, y = slide_all[idx], y_all[idx]
            b = np.digitize(sd_all[idx], edges)
            cell = b * n_slides + slide  # (bin, slide)
            size = nb * n_slides
            for model in all_models(config):
                s = data["scores"][model][idx]
                hit = s >= data["thresholds"][model]
                car, ben = y == 1, y == 0
                tp = np.bincount(cell[car & hit], minlength=size).reshape(nb, n_slides)
                p = np.bincount(cell[car], minlength=size).reshape(nb, n_slides)
                fp = np.bincount(cell[ben & hit], minlength=size).reshape(nb, n_slides)
                nn = np.bincount(cell[ben], minlength=size).reshape(nb, n_slides)
                med = pd.Series(s).groupby(b).median()
                med_c = pd.Series(s[car]).groupby(b[car]).median()
                med_b = pd.Series(s[ben]).groupby(b[ben]).median()
                for k in range(nb):
                    tot = p[k].sum() + nn[k].sum()
                    if tot == 0:
                        continue
                    rlo, rhi = (
                        boot.ratio(tp[k], p[k]) if p[k].sum() else (np.nan, np.nan)
                    )
                    flo, fhi = (
                        boot.ratio(fp[k], nn[k]) if nn[k].sum() else (np.nan, np.nan)
                    )
                    lo = -np.inf if k == 0 else float(edges[k - 1])
                    hi = np.inf if k == nb - 1 else float(edges[k])
                    rows.append(
                        {
                            "boundary": "focus" if boundary == primary else "gland",
                            "boundary_closing_um": boundary,
                            "foci": restriction,
                            "model": model,
                            "bin_lo_um": lo,
                            "bin_hi_um": hi,
                            "bin_mid_um": (lo + hi) / 2 if 0 < k < nb - 1 else np.nan,
                            "n_nuclei": int(tot),
                            "n_carcinoma": int(p[k].sum()),
                            "n_noncarcinoma": int(nn[k].sum()),
                            "n_slides": int(((p[k] + nn[k]) > 0).sum()),
                            "median_score": float(med.get(k, np.nan)),
                            "median_score_carcinoma": float(med_c.get(k, np.nan)),
                            "median_score_noncarcinoma": float(med_b.get(k, np.nan)),
                            "recall": float(tp[k].sum() / p[k].sum())
                            if p[k].sum()
                            else np.nan,
                            "recall_ci_lo": rlo,
                            "recall_ci_hi": rhi,
                            "FPR": float(fp[k].sum() / nn[k].sum())
                            if nn[k].sum()
                            else np.nan,
                            "FPR_ci_lo": flo,
                            "FPR_ci_hi": fhi,
                        }
                    )
    return pd.DataFrame(rows)


def size_classes(config: DictConfig, data: dict) -> NDArray[np.int8]:
    """Per nucleus: 1 / 2 when its (nearest) focus at the primary closing is small / large."""
    a, primary = config.analysis, int(config.geometry.primary_closing_um)
    f = data["foci"][data["foci"]["closing_um"] == primary]
    n = data["nuclei"]
    m = n[["slide", f"focus_{primary}"]].merge(
        f[["slide", "focus", "area_mm2"]].rename(columns={"focus": f"focus_{primary}"}),
        on=["slide", f"focus_{primary}"],
        how="left",
    )
    area = m["area_mm2"].to_numpy()
    return np.select(
        [area < float(a.small_focus_mm2), area > float(a.large_focus_mm2)], [1, 2], 0
    ).astype(np.int8)


def d2_ramp(
    config: DictConfig, data: dict, boot: SlideBootstrap, size_class: NDArray
) -> pd.DataFrame:
    """Deep − edge recall inside foci and near − far FPR outside, with slide-bootstrap CIs."""
    r = config.analysis.ramp
    n = data["nuclei"]
    primary = int(config.geometry.primary_closing_um)
    n_slides = len(data["meta"])
    pos = np.zeros(n_slides, bool)
    pos[data["positive_slides"]] = True
    slide, y = n["slide"].to_numpy(), n["y"].to_numpy()
    rows = []
    for boundary in (primary, 0):
        sd = n[f"sd_{boundary}"].to_numpy()
        base = pos[slide] & np.isfinite(sd)
        bands = {
            "edge": (y == 1) & (sd >= r.edge[0]) & (sd < r.edge[1]),
            "deep": (y == 1) & (sd >= r.deep[0]),
            "near": (y == 0) & (sd >= r.near[0]) & (sd < r.near[1]),
            "far": (y == 0) & (sd < r.far[1]),
        }
        for restriction, code in (("all", None), ("small", 1), ("large", 2)):
            sel = base if code is None else base & (size_class == code)
            for model in all_models(config):
                s = data["scores"][model]
                hit = s >= data["thresholds"][model]
                sums = {}
                for k, m in bands.items():
                    m = m & sel
                    sums[k] = (
                        np.bincount(slide[m & hit], minlength=n_slides).astype(float),
                        np.bincount(slide[m], minlength=n_slides).astype(float),
                        float(np.median(s[m])) if m.any() else np.nan,
                    )
                row = {
                    "boundary": "focus" if boundary == primary else "gland",
                    "foci": restriction,
                    "model": model,
                }
                for (a_, b_), name in (
                    (("deep", "edge"), "recall"),
                    (("near", "far"), "FPR"),
                ):
                    (ha, na, ma), (hb, nb_, mb) = sums[a_], sums[b_]
                    min_n = int(config.analysis.ramp.min_nuclei)
                    ra = ha.sum() / na.sum() if na.sum() >= min_n else np.nan
                    rb = hb.sum() / nb_.sum() if nb_.sum() >= min_n else np.nan
                    with np.errstate(invalid="ignore", divide="ignore"):
                        reps = (boot.counts @ ha) / (boot.counts @ na) - (
                            boot.counts @ hb
                        ) / (boot.counts @ nb_)
                    lo, hi = boot.ci(reps) if np.isfinite(ra - rb) else (np.nan, np.nan)
                    row |= {
                        f"{name}_{a_}": ra,
                        f"{name}_{b_}": rb,
                        f"{name}_{a_}_minus_{b_}": ra - rb,
                        f"{name}_diff_ci_lo": lo,
                        f"{name}_diff_ci_hi": hi,
                        f"n_{a_}": int(na.sum()),
                        f"n_{b_}": int(nb_.sum()),
                        f"median_score_{a_}": ma,
                        f"median_score_{b_}": mb,
                    }
                rows.append(row)
    return pd.DataFrame(rows)


def d3(config: DictConfig, data: dict, boot: SlideBootstrap) -> dict[str, pd.DataFrame]:
    a = config.analysis
    n = data["nuclei"]
    n_slides = len(data["meta"])
    slide, y, cls = n["slide"].to_numpy(), n["y"].to_numpy(), n["cls"].to_numpy()
    pos = np.zeros(n_slides, bool)
    pos[data["positive_slides"]] = True
    on_pos = pos[slide]

    # (a) recall per pattern, FPR per non-carcinoma class (positive and negative slides)
    rate_rows = []
    groups = [
        *[(f"G{p}", "positive slides", (cls == p) & (y == 1)) for p in PATTERNS],
        ("carcinoma", "positive slides", y == 1),
        *[
            (
                CLS_NAMES[c],
                where,
                (cls == c)
                & (y == 0)
                & (on_pos if where == "positive slides" else ~on_pos),
            )
            for c in (2, 1, 0)
            for where in ("positive slides", "negative slides")
        ],
        ("non-carcinoma", "positive slides", (y == 0) & on_pos),
        ("non-carcinoma", "negative slides", (y == 0) & ~on_pos),
    ]
    for model in all_models(config):
        s = data["scores"][model]
        hit = s >= data["thresholds"][model]
        for name, where, m in groups:
            if not m.any():
                continue
            h = np.bincount(slide[m & hit], minlength=n_slides).astype(float)
            t = np.bincount(slide[m], minlength=n_slides).astype(float)
            lo, hi = (
                boot.ratio(h, t) if where == "positive slides" else (np.nan, np.nan)
            )
            rate_rows.append(
                {
                    "model": model,
                    "class": name,
                    "slides": where,
                    "rate": "recall"
                    if name.startswith("G") or name == "carcinoma"
                    else "FPR",
                    "n_nuclei": int(m.sum()),
                    "n_slides": int((t > 0).sum()),
                    "value": float(h.sum() / t.sum()),
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "median_score": float(np.median(s[m])),
                }
            )

    # (b) within-slide AUROC: pattern nuclei vs the slide's non-carcinoma (or benign epithelium) nuclei
    order = np.argsort(slide, kind="stable")
    bounds = np.searchsorted(slide[order], np.arange(n_slides + 1))
    auc_rows = []
    for sl in data["positive_slides"]:
        idx = order[bounds[sl] : bounds[sl + 1]]
        c, yy = cls[idx], y[idx]
        refs = {"non-carcinoma": yy == 0, "benign epithelium": (yy == 0) & (c == 2)}
        for p in PATTERNS:
            mp_ = (yy == 1) & (c == p)
            if mp_.sum() < int(a.min_pattern_nuclei):
                continue
            for ref, mr in refs.items():
                if mr.sum() < int(a.min_benign_nuclei):
                    continue
                for model in all_models(config):
                    s = data["scores"][model][idx]
                    auc_rows.append(
                        {
                            "slide": int(sl),
                            "pattern": f"G{p}",
                            "reference": ref,
                            "model": model,
                            "n_pattern": int(mp_.sum()),
                            "n_reference": int(mr.sum()),
                            "AUROC": auroc(s[mp_], s[mr]),
                        }
                    )
    auc = pd.DataFrame(auc_rows)
    auc_sum = []
    for (ref, model, pat), d in auc.groupby(["reference", "model", "pattern"]):
        q1, med, q3 = np.quantile(d["AUROC"], [0.25, 0.5, 0.75])
        lo, hi = boot.grouped(d["slide"].to_numpy(), np.median, d["AUROC"].to_numpy())
        auc_sum.append(
            {
                "reference": ref,
                "model": model,
                "pattern": pat,
                "n_slides": len(d),
                "median_AUROC": float(med),
                "q1": float(q1),
                "q3": float(q3),
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )
    # paired G3 − G4 on slides with both
    for (ref, model), d in auc.groupby(["reference", "model"]):
        w = d.pivot(index="slide", columns="pattern", values="AUROC")
        if not {"G3", "G4"} <= set(w.columns):
            continue
        w = w.dropna(subset=["G3", "G4"])
        if len(w) < 3:
            continue
        diff = (w["G3"] - w["G4"]).to_numpy()
        lo, hi = boot.grouped(w.index.to_numpy(), np.median, diff)
        auc_sum.append(
            {
                "reference": ref,
                "model": model,
                "pattern": "G3 - G4 (paired)",
                "n_slides": len(w),
                "median_AUROC": float(np.median(diff)),
                "q1": float(np.quantile(diff, 0.25)),
                "q3": float(np.quantile(diff, 0.75)),
                "ci_lo": lo,
                "ci_hi": hi,
            }
        )

    # (d) per-slide recall vs carcinoma share, controlling for the G3 fraction of the carcinoma
    ps = data["per_slide"]
    sl = data["positive_slides"]
    share = (ps["n_carcinoma"] / ps["n"]).to_numpy()[sl]
    g3 = (ps["n_g3"] / ps["n_carcinoma"]).to_numpy()[sl]
    ncar = ps["n_carcinoma"].to_numpy()[sl]
    car = y == 1
    part_rows = []
    for model in all_models(config):
        s = data["scores"][model]
        tp = np.bincount(
            slide[car & (s >= data["thresholds"][model])], minlength=n_slides
        )[sl]
        recall = tp / ncar
        for min_n in (1, int(a.min_pattern_nuclei)):
            k = ncar >= min_n
            x, r, z, sk = share[k], recall[k], g3[k], sl[k]
            for stat, fn, args in (
                ("spearman(share, recall)", spearman, (x, r)),
                ("partial(share, recall | G3 fraction)", partial_spearman, (x, r, z)),
                ("spearman(G3 fraction, recall)", spearman, (z, r)),
                ("partial(G3 fraction, recall | share)", partial_spearman, (z, r, x)),
                ("spearman(share, G3 fraction)", spearman, (x, z)),
            ):
                lo, hi = boot.grouped(sk, fn, *args)
                part_rows.append(
                    {
                        "model": model,
                        "min_carcinoma_nuclei": min_n,
                        "statistic": stat,
                        "n_slides": int(k.sum()),
                        "rho": fn(*args),
                        "ci_lo": lo,
                        "ci_hi": hi,
                    }
                )
    return {
        "rates": pd.DataFrame(rate_rows),
        "auroc_slides": auc,
        "auroc": pd.DataFrame(auc_sum),
        "partial": pd.DataFrame(part_rows),
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def ci(r: pd.Series, v: str = "spearman_rho", f: str = "+.2f") -> str:
    return f"{r[v]:{f}} [{r.ci_lo:{f}}, {r.ci_hi:{f}}]"


def report(config: DictConfig, checks: dict, t: dict[str, pd.DataFrame]) -> str:
    models = list(config.models)
    lab = {k: v.short for k, v in all_models(config).items()}
    primary = int(config.geometry.primary_closing_um)
    th = checks["thresholds"]
    L = [
        "# Focus size, boundary profile and Gleason pattern: PANDA Radboud test",
        "",
        (
            f"{checks['test']['positive_slides']} positive / {checks['test']['negative_slides']} negative test slides, "
            f"{checks['test']['nuclei']:,} nuclei ({checks['test']['carcinoma_nuclei']:,} carcinoma). Score = seed-mean "
            "probability (3 seeds); thresholds = Youden on each model's seed-mean validation probability. CIs: "
            f"percentile, {config.bootstrap.n_resamples} resamples of the positive slides."
        ),
        "",
        "## Models and thresholds",
        "",
        "| Model | Evaluation runs | Training runs | Youden t | Val. recall | Val. FPR | Re-prediction max abs. diff |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in [*models, *config.sensitivity_models]:
        r = th[name]
        L.append(
            f"| {all_models(config)[name].label} | {', '.join(x['eval_run'][:8] for x in r['runs'])} | "
            f"{', '.join(x['training_run'][:8] for x in r['runs'])} | {r['threshold']:.4f} | "
            f"{r['validation']['recall']:.3f} | {r['validation']['FPR']:.3f} | "
            f"{max(x['max_abs_diff'] for x in r['rerun_test_sample']):.3f} |"
        )
    L += [
        "",
        f"Main model: Youden {th['shape_knn']['threshold']:.4f} (thesis figure: {config.models.shape_knn.thesis_threshold}).",
        "",
        f"## D1 Focus size (closing {primary} µm primary; foci with ≥ {config.analysis.min_focus_nuclei} carcinoma nuclei)",
        "",
        "Spearman ρ of focus area with the focus median score and with the focus recall [95 % CI].",
        "",
        "| Model | Closing | Foci | ρ (median score) | ρ (recall) |",
        "| --- | --- | --- | --- | --- |",
    ]
    c = t["d1_spearman"]
    for name in models:
        for d in config.geometry.closings_um:
            q = c[(c.model == name) & (c.closing_um == int(d)) & (c.foci == "all")]
            ms, rc = (
                q[q.variable == "median_score"].iloc[0],
                q[q.variable == "recall"].iloc[0],
            )
            L.append(f"| {lab[name]} | {d} µm | {ms.n_foci:,} | {ci(ms)} | {ci(rc)} |")
    b = t["d1_bins"]
    b = b[(b.closing_um == primary) & (b.foci == "all")]
    bins = list(dict.fromkeys(b["area_bin_mm2"]))
    L += [
        "",
        f"Per area bin (mm², closing {primary} µm): median of the focus median scores / pooled recall.",
        "",
        "| Model | " + " | ".join(bins) + " |",
        "| --- |" + " --- |" * len(bins),
        "| Foci (main model) | "
        + " | ".join(
            str(int(b[(b.model == "shape_knn") & (b.area_bin_mm2 == x)].n_foci.sum()))
            for x in bins
        )
        + " |",
    ]
    for name in models:
        cells = []
        for x in bins:
            q = b[(b.model == name) & (b.area_bin_mm2 == x)]
            cells.append(
                f"{q.median_score_median.iloc[0]:.2f} / {q.recall_pooled.iloc[0]:.2f}"
                if len(q)
                else "–"
            )
        L.append(f"| {lab[name]} | " + " | ".join(cells) + " |")

    rp = t["d2_ramp"]
    rc = config.analysis.ramp
    L += [
        "",
        "## D2 Boundary profile",
        "",
        (
            f"Signed distance of the centroid to the boundary of the {primary} µm-closed carcinoma mask (focus) or of the "
            "unclosed mask (gland). Ramp: recall of carcinoma nuclei ≥ "
            f"{rc.deep[0]} µm inside minus recall within {rc.edge[0]}–{rc.edge[1]} µm of the boundary; FPR of "
            f"non-carcinoma nuclei within {-rc.near[0]} µm outside minus FPR > {-rc.far[1]} µm outside."
        ),
        "",
        "| Boundary | Foci | Model | Recall edge | Recall deep | Deep − edge [CI] | FPR near | FPR far | Near − far [CI] |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, r in rp[rp.model.isin(models) & (rp.boundary == "focus")].iterrows():
        L.append(
            f"| {r.boundary} | {r.foci} | {lab[r.model]} | {r.recall_edge:.3f} | {r.recall_deep:.3f} | "
            f"{r.recall_deep_minus_edge:+.3f} [{r.recall_diff_ci_lo:+.3f}, {r.recall_diff_ci_hi:+.3f}] | "
            f"{r.FPR_near:.3f} | {r.FPR_far:.3f} | {r.FPR_near_minus_far:+.3f} [{r.FPR_diff_ci_lo:+.3f}, {r.FPR_diff_ci_hi:+.3f}] |"
        )

    rates = t["d3_rates"]
    L += [
        "",
        "## D3 Gleason pattern",
        "",
        "Recall per pattern (positive slides) and FPR per non-carcinoma class [95 % CI].",
        "",
        "| Model | G3 | G4 | G5 | Benign epith. (pos.) | Stroma (pos.) | Non-carc. (neg. slides) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in models:
        q = rates[rates.model == name].set_index(["class", "slides"])

        def cell(k: str, w: str = "positive slides", q: pd.DataFrame = q) -> str:
            if (k, w) not in q.index:
                return "–"
            r = q.loc[(k, w)]
            return f"{r.value:.3f}" + (
                f" [{r.ci_lo:.3f}, {r.ci_hi:.3f}]" if np.isfinite(r.ci_lo) else ""
            )

        L.append(
            f"| {lab[name]} | {cell('G3')} | {cell('G4')} | {cell('G5')} | {cell('benign epithelium')} | "
            f"{cell('stroma')} | {cell('non-carcinoma', 'negative slides')} |"
        )
    n3 = rates[(rates.model == "shape_knn") & rates["class"].isin(["G3", "G4", "G5"])]
    L += [
        "",
        "Carcinoma nuclei per pattern: "
        + ", ".join(f"{r['class']} {r.n_nuclei:,}" for _, r in n3.iterrows())
        + ".",
    ]
    au = t["d3_auroc"]
    for ref in ("non-carcinoma", "benign epithelium"):
        a = au[au.reference == ref]
        L += [
            "",
            (
                f"Within-slide AUROC, pattern vs the slide's {ref} nuclei (median [IQR] over slides with ≥ "
                f"{config.analysis.min_pattern_nuclei} nuclei of the pattern and ≥ {config.analysis.min_benign_nuclei} reference nuclei; "
                "paired G3 − G4 on slides with both, median [95 % CI])."
            ),
            "",
            "| Model | G3 | G4 | G5 | G3 − G4 (paired) |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name in models:
            q = a[a.model == name].set_index("pattern")
            cells = []
            for p in ("G3", "G4", "G5"):
                cells.append(
                    f"{q.loc[p].median_AUROC:.3f} [{q.loc[p].q1:.3f}, {q.loc[p].q3:.3f}] (n = {q.loc[p].n_slides})"
                    if p in q.index
                    else "–"
                )
            k = "G3 - G4 (paired)"
            cells.append(
                f"{q.loc[k].median_AUROC:+.3f} [{q.loc[k].ci_lo:+.3f}, {q.loc[k].ci_hi:+.3f}] (n = {q.loc[k].n_slides})"
                if k in q.index
                else "–"
            )
            L.append(f"| {lab[name]} | " + " | ".join(cells) + " |")
    L += [
        "",
        f"Focus size within dominant pattern (closing {primary} µm): Spearman ρ of area with median score / recall.",
        "",
        "| Model | G3-dominant | G3-pure | G4-dominant | G5-dominant |",
        "| --- | --- | --- | --- | --- |",
    ]
    c = t["d1_spearman"]
    for name in models:
        cells = []
        for f_ in ("G3-dominant", "G3-pure", "G4-dominant", "G5-dominant"):
            q = c[(c.model == name) & (c.closing_um == primary) & (c.foci == f_)]
            if q.empty:
                cells.append("–")
                continue
            ms, rc_ = (
                q[q.variable == "median_score"].iloc[0],
                q[q.variable == "recall"].iloc[0],
            )
            cells.append(f"{ci(ms)} / {ci(rc_)} (n = {ms.n_foci})")
        L.append(f"| {lab[name]} | " + " | ".join(cells) + " |")
    pt = t["d3_partial"]
    pt = pt[pt.min_carcinoma_nuclei == 1]
    stats = list(dict.fromkeys(pt["statistic"]))
    L += [
        "",
        "Per positive slide: recall vs carcinoma share (carcinoma nuclei / nuclei) and the G3 fraction of the carcinoma.",
        "",
        "| Model | " + " | ".join(stats) + " |",
        "| --- |" + " --- |" * len(stats),
    ]
    for name in models:
        q = pt[pt.model == name].set_index("statistic")
        L.append(
            f"| {lab[name]} | " + " | ".join(ci(q.loc[s], "rho") for s in stats) + " |"
        )

    s = t["sensitivity"]
    L += [
        "",
        "## Sensitivity: blank token with the Q2/Q3 seeds",
        "",
        s,
        "",
        "## Checks",
        "",
    ]
    L += [f"- {k}: {v}" for k, v in checks["notes"].items()]
    L.append("")
    return "\n".join(L)


def sensitivity_text(config: DictConfig, t: dict[str, pd.DataFrame]) -> str:
    primary = int(config.geometry.primary_closing_um)
    c = t["d1_spearman"]
    rp = t["d2_ramp"]
    rates = t["d3_rates"]
    out = []
    for name in ("blank_knn", "blank_knn_q2"):
        if name not in all_models(config):
            continue
        q = c[(c.model == name) & (c.closing_um == primary) & (c.foci == "all")]
        ms, rc = (
            q[q.variable == "median_score"].iloc[0],
            q[q.variable == "recall"].iloc[0],
        )
        r = rp[(rp.model == name) & (rp.boundary == "focus") & (rp.foci == "all")].iloc[
            0
        ]
        g = rates[
            (rates.model == name) & (rates.slides == "positive slides")
        ].set_index("class")["value"]
        out.append(
            f"- {all_models(config)[name].label}: D1 ρ {ci(ms)} (score), {ci(rc)} (recall); D2 deep − edge recall "
            f"{r.recall_deep_minus_edge:+.3f}; recall G3 / G4 / G5 {g['G3']:.3f} / {g['G4']:.3f} / {g['G5']:.3f}."
        )
    return "\n".join(out)


# --------------------------------------------------------------------------
# Analysis stage, logging
# --------------------------------------------------------------------------


def stage_analysis(config: DictConfig) -> tuple[Path, dict, dict[str, pd.DataFrame]]:
    from visualization.focus_size_figures import save_figures

    out = Path(config.out_dir)
    thresholds = json.loads((out / "thresholds.json").read_text())
    assert set(all_models(config)) <= set(thresholds), "run the validation stage first"
    data = assemble(config, thresholds)
    boot = SlideBootstrap(config, len(data["meta"]), data["positive_slides"])
    d1_corr, d1_bins, per_focus = d1(config, data, boot)
    size_class = size_classes(config, data)
    tables = {
        "d1_spearman": d1_corr,
        "d1_bins": d1_bins,
        "d2_profile": d2(config, data, boot, size_class),
        "d2_ramp": d2_ramp(config, data, boot, size_class),
    }
    tables |= {f"d3_{k}": v for k, v in d3(config, data, boot).items()}
    tables["sensitivity"] = sensitivity_text(config, tables)

    for k, v in tables.items():
        if isinstance(v, pd.DataFrame):
            v.to_csv(out / f"{k}.csv", index=False)
    for (model, closing), fs in per_focus.items():
        fs.drop(columns=["focus_index"]).to_csv(
            out / f"d1_foci_{model}_c{closing}.csv.gz", index=False
        )
    per_model = out / "per_model"
    per_model.mkdir(exist_ok=True)
    for model in all_models(config):
        for k, v in tables.items():
            if isinstance(v, pd.DataFrame) and "model" in v.columns:
                v[v.model == model].to_csv(per_model / f"{k}_{model}.csv", index=False)

    n = data["nuclei"]
    ps = data["per_slide"]
    pos = data["positive_slides"]
    primary = int(config.geometry.primary_closing_um)
    foci_p = per_focus[("shape_knn", primary)]
    checks = {
        "thresholds": thresholds,
        "seeds": data["seeds"],
        "test": {
            "slides": len(data["meta"]),
            "positive_slides": len(pos),
            "negative_slides": len(data["meta"]) - len(pos),
            "nuclei": len(n),
            "carcinoma_nuclei": int(n["y"].sum()),
        },
        "rule_vs_label": data["rule_vs_label"],
        "classes": {
            CLS_NAMES[int(k)]: int(v)
            for k, v in n.loc[n["y"] == 0, "cls"].value_counts().items()
        },
        "patterns": {
            CLS_NAMES[int(k)]: int(v)
            for k, v in n.loc[n["y"] == 1, "cls"].value_counts().items()
        },
        "foci": {
            int(d): len(per_focus[("shape_knn", int(d))])
            for d in config.geometry.closings_um
        },
        "notes": {
            "labels": f"vertex rule on the Radboud label masks reproduces {config.label_column} on every carcinoma slide "
            f"({data['rule_vs_label']['carcinoma_slides_mismatch']} mismatches); "
            f"{data['rule_vs_label']['negative_slides_rule_carcinoma']} nuclei of is_carcinoma = False slides lie in "
            "pattern ≥ 3 regions (label 0 by the slide rule, class 'other' here)",
            "foci": f"closing {primary} µm: {len(foci_p):,} foci with carcinoma nuclei, "
            f"{int((foci_p.n_carcinoma >= int(config.analysis.min_focus_nuclei)).sum()):,} with ≥ "
            f"{config.analysis.min_focus_nuclei}",
            "pixel": f"geometry on the label mask subsampled ×{config.geometry.stride} "
            f"({float(data['meta']['mpp_x'].iloc[0]) * int(config.geometry.stride):.3f} µm per pixel, slide-table mpp)",
            "slides": "slides are referred to by their row in the test table sorted by slide id; no ids in the tables",
            "G3 fraction": f"positive slides with G3 carcinoma nuclei: {int((ps['n_g3'].to_numpy()[pos] > 0).sum())}",
        },
    }
    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=str))
    (out / "report.md").write_text(report(config, checks, tables))
    save_figures(config, tables, per_focus, out / "figures", data["thresholds"])
    return out, checks, tables


def preflight(out: Path, allowed: set[str]) -> None:
    """No credentials; every 32-hex token is one of the runs (no slide ids)."""
    for path in [*out.glob("*.*"), *(out / "per_model").glob("*.csv")]:
        if path.suffix not in {".csv", ".json", ".md", ".yaml"}:
            continue
        text = path.read_text()
        assert not CREDENTIAL_URL.search(text), f"credential URL in {path.name}"
        unknown = set(HEX32.findall(text)) - allowed
        assert not unknown, f"{len(unknown)} unknown ids in {path.name}"


def log_run(config: DictConfig, client: MlflowClient, out: Path, checks: dict) -> str:
    th = checks["thresholds"]
    params = {
        "dataset": config.dataset_label,
        "primary_closing_um": config.geometry.primary_closing_um,
        "closings_um": ",".join(str(d) for d in config.geometry.closings_um),
        "n_resamples": config.bootstrap.n_resamples,
        **{f"threshold_{k}": round(v["threshold"], 6) for k, v in th.items()},
        **{
            f"eval_runs_{k}": ",".join(m.eval_runs)
            for k, m in all_models(config).items()
        },
    }
    tags = {
        "mlflow.runName": config.metadata.run_name,
        "mlflow.note.content": config.metadata.description,
        **git_info(*CODE),
    }
    for text in [*map(str, tags.values()), *map(str, params.values())]:
        assert not CREDENTIAL_URL.search(text), "credential URL in a tag or param"
    run_id = client.create_run(
        str(config.metadata.experiment_id), run_name=config.metadata.run_name, tags=tags
    ).info.run_id
    user = client.get_run(config.models.shape_knn.eval_runs[0]).data.tags.get(
        "mlflow.user"
    )
    if user:
        client.set_tag(run_id, "mlflow.user", user)
    for k, v in params.items():
        client.log_param(run_id, k, v)
    for path in sorted(out.glob("*.*")):
        client.log_artifact(run_id, str(path))
    client.log_artifacts(run_id, str(out / "per_model"), "per_model")
    client.log_artifacts(run_id, str(out / "figures"), "figures")
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
    return run_id


@with_cli_args(["+postprocessing=focus_size_analysis_radboud"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
def main(config: DictConfig) -> None:
    # download_artifacts resolves mlflow-artifacts:/ against the environment variable
    os.environ["MLFLOW_TRACKING_URI"] = str(config.mlflow_tracking_uri)
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    client = MlflowClient()
    stages = set(config.stages)
    if "validation" in stages:
        stage_validation(config, client)
    if "geometry" in stages:
        checks = stage_geometry(config)
        print(checks.drop(columns=["mask_shape", "scale"]).describe().T.to_string())
    if "analysis" not in stages:
        return
    out, checks, _ = stage_analysis(config)
    (out / "config-resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    allowed = {e for m in all_models(config).values() for e in m.eval_runs}
    allowed |= {
        r["training_run"] for v in checks["thresholds"].values() for r in v["runs"]
    }
    allowed |= set(HEX32.findall(OmegaConf.to_yaml(config, resolve=True)))
    preflight(out, allowed)
    if config.log_mlflow:
        run_id = log_run(config, client, out, checks)
        print(
            f"MLflow run {run_id} (experiment {config.metadata.experiment_id}); outputs in {out}"
        )
    else:
        print(f"log_mlflow=false: outputs in {out}")


if __name__ == "__main__":
    main()
