# ruff: noqa: RUF002  (Greek letters / minus signs in report text are intended)
r"""Study small_foci: interventions against context dilution on small carcinoma foci (PANDA Radboud).

Focus analysis 7287d498 (MLflow run 79bfa6dd): in the smallest foci the k-NN model has lower recall
than the self-only model. This script holds the study's data preparation and its shared evaluation.

Foci (as 7287d498): 8-connected components of the Gleason 3-5 mask of the Radboud label masks after a
closing with a 50 µm disk (``postprocessing.focus_geometry_radboud``). µm per pixel = the slide
tables' ``mpp_x`` (0.4862, the value in the Radboud TIFF headers) × the mask stride.

Stages (``stages``):

``geometry``
    Per-nucleus focus and signed boundary distance of every train-table slide (train + validation),
    as the test geometry of 7287d498.
``foci_table``
    Training input of A1 (focus-balanced seeding) and A2 (focus-size loss weight): the carcinoma
    nuclei (vertex rule = ``annot_label``, checked per slide) of the train-table slides with their
    position in the id-sorted nuclei, focus and focus size in carcinoma nuclei. Slides are keyed by
    ``slide_key`` (``nuclei_graph.data.foci.slide_key``), not by id. Logged to MLflow (experiment 37).
``predict``
    Validation (or test) predictions of the study's training runs with their own commit and config.
``evaluate`` / ``fusion`` / ``test`` / ``report``
    Shared evaluation, step 0 (late fusion), step 3 (bootstrap on test) and the report.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from nuclei_graph.data.datamodules.base import split_train_val
from nuclei_graph.data.foci import FOCI_COLUMNS, slide_key
from postprocessing.bootstrap_sources import (
    bin_logits,
    bootstrap_p,
    hist_metrics,
    holm,
    label_predictions,
    load_labels,
    load_run,
)
from postprocessing.focus_geometry_radboud import slide_geometry
from postprocessing.nucleus_results_mmci import CREDENTIAL_URL
from postprocessing.patient_error_analysis_mmci import calibrated_ap, calibration
from postprocessing.patient_score_quantiles_mmci import youden_threshold
from postprocessing.tile.bootstrap_pixel_vs_nuclei import git_info


REPO = Path(__file__).parents[1]
INFERENCE = "postprocessing/nucleus_inference_at_commit.py"
HEX32 = re.compile(r"(?<![0-9a-f])[0-9a-f]{32}(?![0-9a-f])")
CKPT_EPOCH = re.compile(r"epoch=(\d+)")
CODE = (
    "postprocessing/small_foci_radboud.py",
    "postprocessing/nucleus_inference_at_commit.py",
    "visualization/small_foci_figures.py",
    "configs/postprocessing/small_foci_radboud.yaml",
)


# --------------------------------------------------------------------------
# Slides
# --------------------------------------------------------------------------


def remap(value, config: DictConfig):
    """Paths of the public cluster's project folder, as mounted here (recursively)."""
    remote, local = str(config.remote_project_path), str(config.local_project_path)
    if isinstance(value, str):
        return local + value[len(remote) :] if value.startswith(remote) else value
    if isinstance(value, dict):
        return {k: remap(v, config) for k, v in value.items()}
    if isinstance(value, list):
        return [remap(v, config) for v in value]
    return value


def read_slides(config: DictConfig, uri: str) -> pd.DataFrame:
    meta = pd.read_parquet(download_artifacts(uri)).sort_values(
        "slide_id", ignore_index=True
    )
    meta["slide_nuclei_path"] = [remap(p, config) for p in meta["slide_nuclei_path"]]
    return meta


def train_table(config: DictConfig) -> pd.DataFrame:
    return read_slides(config, config.train_metadata_uri)


def validation_slides(config: DictConfig) -> pd.DataFrame:
    s = config.validation_split
    _, val = split_train_val(
        read_slides(config, config.train_metadata_uri),
        float(s.size),
        s.stratify_col,
        s.group_col,
    )
    return val.sort_values("slide_id", ignore_index=True)


def test_slides(config: DictConfig) -> pd.DataFrame:
    return read_slides(config, config.test_metadata_uri)


# --------------------------------------------------------------------------
# Geometry and the training focus table
# --------------------------------------------------------------------------


def stage_geometry(config: DictConfig) -> pd.DataFrame:
    meta = train_table(config)
    g = config.geometry
    out_dir = Path(config.geometry_dirs.train)
    jobs = [
        {
            "slide": s,
            "slide_id": row.slide_id,
            "out_dir": str(out_dir),
            "nuclei_path": row.slide_nuclei_path,
            "mask_path": str(
                Path(config.label_masks_dir) / f"{row.slide_id}_mask.tiff"
            ),
            "slide_path": row.slide_path,
            "label_path": str(Path(config.supervision_dir) / f"{row.slide_id}.parquet")
            if row.is_carcinoma
            else None,
            "label_column": config.label_column,
            "overlap_threshold": float(config.overlap_threshold),
            "stride": int(g.stride),
            "mpp": float(row.mpp_x),
            "closings_um": [int(g.closing_um)],
        }
        for s, row in enumerate(meta.itertuples())
    ]
    missing = [j["slide"] for j in jobs if not Path(j["mask_path"]).exists()]
    assert not missing, f"{len(missing)} train-table slides without a label mask"
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
    print(
        f"geometry: {len(checks)} slides, {int(checks['nuclei'].sum()):,} nuclei, "
        f"{int(checks['rule_carcinoma'].sum()):,} carcinoma nuclei"
    )
    return checks


def build_foci_table(config: DictConfig) -> tuple[pd.DataFrame, dict]:
    """Carcinoma nuclei of the train-table slides: slide_key, index, key, focus, focus_nuclei."""
    meta = train_table(config)
    d = int(config.geometry.closing_um)
    gdir = Path(config.geometry_dirs.train)
    parts, stats = [], {"slides": 0, "carcinoma_slides_without_foci": 0}
    for row in tqdm(meta.itertuples(), total=len(meta), desc="foci table"):
        stats["slides"] += 1
        n = pd.read_parquet(
            gdir / "nuclei" / f"{row.slide_id}.parquet",
            columns=["key", "y_rule", f"focus_{d}"],
        )
        # geometry rows are in id order (nuclei sorted by id), as the training dataset's nuclei
        assert len(n) == row.nuclei_count, f"slide {row.Index}: nuclei count differs"
        car = np.flatnonzero(n["y_rule"].to_numpy() == 1)
        if not row.is_carcinoma:
            continue  # labels are 0 on negative slides, whatever the mask says
        if len(car) == 0:
            stats["carcinoma_slides_without_foci"] += 1
            continue
        focus = n[f"focus_{d}"].to_numpy()[car]
        assert (focus >= 1).all(), f"slide {row.Index}: carcinoma nucleus without focus"
        _, inverse, counts = np.unique(focus, return_inverse=True, return_counts=True)
        parts.append(
            pd.DataFrame(
                {
                    "slide_key": np.full(len(car), slide_key(row.slide_id), np.uint64),
                    "slide_nuclei": np.full(len(car), len(n), np.int32),
                    "index": car.astype(np.int32),
                    "key": n["key"].to_numpy()[car],
                    "focus": focus.astype(np.int32),
                    "focus_nuclei": counts[inverse].astype(np.int32),
                }
            )
        )
    table = pd.concat(parts, ignore_index=True)[list(FOCI_COLUMNS)]
    assert table["slide_key"].nunique() == len(parts), "slide_key collision"
    foci = table.drop_duplicates(["slide_key", "focus"])
    stats |= {
        "carcinoma_slides": len(parts),
        "carcinoma_nuclei": len(table),
        "foci": len(foci),
        "focus_nuclei_median": float(foci["focus_nuclei"].median()),
        "foci_per_slide_median": float(foci.groupby("slide_key").size().median()),
        "closing_um": d,
    }
    return table, stats


def stage_foci_table(config: DictConfig) -> None:
    table, stats = build_foci_table(config)
    path = Path(config.foci_table.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    (path.parent / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    if not config.foci_table.log_mlflow:
        return
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    with mlflow.start_run(
        experiment_id=str(config.metadata.experiment_id),
        run_name=str(config.foci_table.run_name),
        tags={"study": "small_foci", "role": "data"},
    ) as run:
        mlflow.log_params(
            {"closing_um": stats["closing_um"], "stride": int(config.geometry.stride)}
        )
        mlflow.log_metrics({k: v for k, v in stats.items() if k != "closing_um"})
        mlflow.log_artifact(str(path), "foci_table")
        mlflow.log_artifact(str(path.parent / "stats.json"), "foci_table")
        print(
            f"focus table: mlflow-artifacts:/{config.metadata.experiment_id}/"
            f"{run.info.run_id}/artifacts/foci_table/{path.name}"
        )


# --------------------------------------------------------------------------
# Split data: nuclei (geometry order = slide, then id), labels, foci
# --------------------------------------------------------------------------


def split_slides(config: DictConfig, split: str) -> pd.DataFrame:
    assert split in ("validation", "test"), split
    return validation_slides(config) if split == "validation" else test_slides(config)


def load_split(config: DictConfig, split: str) -> dict:
    """Nuclei of the split's slides (sorted by slide row, then key) with labels and geometry."""
    slides = split_slides(config, split)
    gdir = Path(
        config.geometry_dirs.train
        if split == "validation"
        else config.geometry_dirs.test
    )
    d = int(config.geometry.closing_um)
    nuc, foci = [], []
    for s, sid in enumerate(tqdm(slides["slide_id"], desc=f"{split} geometry")):
        n = pd.read_parquet(
            gdir / "nuclei" / f"{sid}.parquet",
            columns=["key", "y_rule", f"focus_{d}", f"sd_{d}"],
        )
        n.insert(0, "slide", np.int16(s))
        nuc.append(n)
        f = pd.read_parquet(gdir / "foci" / f"{sid}.parquet")
        f = f[f["closing_um"] == d]
        foci.append(
            pd.DataFrame({"slide": s, "focus": f["focus"], "area_mm2": f["area_mm2"]})
        )
    nuclei = pd.concat(nuc, ignore_index=True).sort_values(
        ["slide", "key"], ignore_index=True
    )
    labels = load_labels(slides, Path(config.supervision_dir), config.label_column)
    carcinoma = slides["is_carcinoma"].to_numpy(bool)
    y, unmatched = label_predictions(nuclei[["slide", "key"]], labels, carcinoma)
    slide = nuclei["slide"].to_numpy()
    mismatch = int(((nuclei["y_rule"].to_numpy() != y) & carcinoma[slide]).sum())
    assert mismatch == 0, (
        f"{split}: vertex rule != {config.label_column} on {mismatch} nuclei"
    )
    return {
        "split": split,
        "slides": slides,
        "slide": slide,
        "key": nuclei["key"].to_numpy(),
        "y": y.astype(np.int8),
        "focus": nuclei[f"focus_{d}"].to_numpy(),
        "sd": nuclei[f"sd_{d}"].to_numpy(),
        "foci": pd.concat(foci, ignore_index=True),
        "unmatched": unmatched,
    }


def eval_index(config: DictConfig, data: dict) -> dict:
    """Per nucleus: focus-size bin (carcinoma nuclei), distance band (benign nuclei); low-prevalence slides."""
    ev = config.evaluation
    y, slide, focus, sd = data["y"], data["slide"], data["focus"], data["sd"]
    n_slides = len(data["slides"])
    car = np.flatnonzero(y == 1)
    gid = slide[car].astype(np.int64) * 10_000_000 + focus[car]
    assert (focus[car] >= 1).all(), "carcinoma nucleus without a focus"
    uniq, inv, cnt = np.unique(gid, return_inverse=True, return_counts=True)
    f = data["foci"]
    area = pd.Series(
        f["area_mm2"].to_numpy(),
        index=f["slide"].to_numpy().astype(np.int64) * 10_000_000
        + f["focus"].to_numpy(),
    )
    focus_area = area.reindex(uniq).to_numpy()
    assert np.isfinite(focus_area).all(), "focus without an area"
    edges = np.asarray(ev.area_bins_mm2, float)
    fbin = (
        np.searchsorted(edges, focus_area, side="right") - 1
    )  # [lo, hi), as pd.cut(right=False)
    fbin[cnt < int(ev.min_focus_nuclei)] = -1
    nbin = np.full(len(y), -1, np.int16)
    nbin[car] = fbin[inv]
    n_per_slide = np.bincount(slide, minlength=n_slides)
    car_per_slide = np.bincount(slide, weights=(y == 1), minlength=n_slides)
    positive = car_per_slide > 0
    band = np.full(len(y), -1, np.int8)
    outside = (y == 0) & positive[slide] & (sd < 0)
    for j, (lo, hi) in enumerate(ev.distance_bands_um):
        band[outside & (-sd >= lo) & (-sd < hi)] = j
    prevalence = car_per_slide / np.maximum(n_per_slide, 1)
    low = np.flatnonzero(
        positive
        & (prevalence < float(ev.low_prevalence))
        & (car_per_slide >= int(ev.min_carcinoma_nuclei))
    )
    starts = np.r_[0, np.cumsum(n_per_slide)]
    return {
        "bin": nbin,
        "band": band,
        "bin_labels": [f"{lo:g}-{hi:g}" for lo, hi in pairwise(edges)],
        "band_labels": [f"{lo:g}-{hi:g}" for lo, hi in ev.distance_bands_um],
        "small_bins": list(range(int(ev.small_bins))),
        "positive_slides": positive,
        "low_slides": low,
        "low_c": {
            int(s): calibration(float(prevalence[s]), float(ev.pi0)) for s in low
        },
        "starts": starts,
        "focus_nuclei_binned": {
            b: int(cnt[fbin == b].sum()) for b in range(len(edges) - 1)
        },
        "foci_binned": {b: int((fbin == b).sum()) for b in range(len(edges) - 1)},
    }


# --------------------------------------------------------------------------
# Scores: logits per model and seed, aligned to the split's nuclei
# --------------------------------------------------------------------------


def to_logit(p: NDArray, clip: float) -> NDArray[np.float32]:
    p64 = np.clip(p.astype(np.float64), 1e-300, 1.0)
    with np.errstate(divide="ignore"):
        z = np.log(p64) - np.log1p(-p64)
    return np.clip(z, -clip, clip).astype(np.float32)


def aligned(data: dict, df: pd.DataFrame, what: str) -> pd.DataFrame:
    df = df.sort_values(["slide", "key"], ignore_index=True)
    assert len(df) == len(data["key"]), (
        f"{what}: {len(df)} nuclei, the split has {len(data['key'])}"
    )
    assert np.array_equal(df["slide"].to_numpy(), data["slide"]), (
        f"{what}: slides differ"
    )
    assert np.array_equal(df["key"].to_numpy(), data["key"]), f"{what}: nuclei differ"
    return df


def existing_scores(
    config: DictConfig, name: str, data: dict
) -> list[NDArray[np.float32]]:
    m = config.existing_models[name]
    clip = float(config.fusion.logit_clip)
    out = []
    for train, eval_run, commit in zip(
        m.training_runs, m.eval_runs, m.eval_commits, strict=True
    ):
        if data["split"] == "validation":
            path = (
                Path(config.existing_inference_cache)
                / commit[:8]
                / f"{train}_validation.parquet"
            )
            df = pd.read_parquet(path)
        else:
            slide_ids = data["slides"]["slide_id"].tolist()
            dirs = [Path(d) for d in config.test_cache_dirs]
            cache = next(
                (
                    d
                    for d in dirs
                    if (d / f"{eval_run}.parquet").exists()
                    and (d / f"{eval_run}.json").exists()
                ),
                dirs[-1],
            )
            df = load_run(eval_run, slide_ids, cache, config.predictions_artifact)[0]
        df = aligned(data, df, f"{name} {train[:8]}")
        out.append(to_logit(df["p"].to_numpy(), clip))
    return out


def study_scores(
    config: DictConfig, name: str, data: dict, component: str | None = None
) -> list[NDArray[np.float32]]:
    col = "logit" if component is None else f"logit_{component}"
    out = []
    for run in config.study_models[name].training_runs:
        path = Path(config.predictions_dir) / f"{run}_{data['split']}.parquet"
        if not path.exists():
            print(
                f"  {name}: no {data['split']} predictions of {run[:8]} (stage predict)"
            )
            continue
        df = aligned(
            data,
            pd.read_parquet(path, columns=["slide", "key", col]),
            f"{name} {run[:8]}",
        )
        out.append(df[col].to_numpy(np.float32))
    return out


def all_scores(
    config: DictConfig, data: dict
) -> dict[str, tuple[str, list[NDArray[np.float32]]]]:
    """{model: (label, logits per seed)}: the existing models, then the study's runs (and B1's components)."""
    scores = {}
    for name, m in config.existing_models.items():
        scores[name] = (m.label, existing_scores(config, name, data))
    for name, m in config.study_models.items():
        z = study_scores(config, name, data)
        if not z:
            continue
        scores[name] = (m.label, z)
        if m.get("components"):
            for c in ("context", "self"):
                scores[f"{name}:{c}"] = (
                    f"{m.label}: {c} logit",
                    study_scores(config, name, data, c),
                )
    return scores


# --------------------------------------------------------------------------
# Shared evaluation
# --------------------------------------------------------------------------


def seed_metrics(data: dict, ix: dict, z: NDArray, t: float) -> tuple[dict, NDArray]:
    """Metrics of one seed's logits `z` at threshold `t`; per-slide calibrated AUPRC of the low slides."""
    y = data["y"]
    hit = z >= t
    pos, neg = y == 1, y == 0
    m = {
        "AUPRC": float(average_precision_score(y, z)),
        "AUROC": float(roc_auc_score(y, z)),
        "threshold_logit": float(t),
        "recall": float(hit[pos].mean()),
        "FPR": float(hit[neg].mean()),
        "FPR_negative_slides": float(
            hit[neg & ~ix["positive_slides"][data["slide"]]].mean()
        ),
    }
    for b, lab in enumerate(ix["bin_labels"]):
        sel = ix["bin"] == b
        m[f"recall_bin_{lab}"] = float(hit[sel].mean()) if sel.any() else np.nan
    small = np.isin(ix["bin"], ix["small_bins"])
    m["recall_small"] = float(hit[small].mean())
    for j, lab in enumerate(ix["band_labels"]):
        sel = ix["band"] == j
        m[f"FPR_out_{lab}um"] = float(hit[sel].mean()) if sel.any() else np.nan
    cal = np.array(
        [
            calibrated_ap(
                y[ix["starts"][s] : ix["starts"][s + 1]],
                z[ix["starts"][s] : ix["starts"][s + 1]],
                ix["low_c"][int(s)],
            )
            for s in ix["low_slides"]
        ]
    )
    return m, cal


def thresholds_path(config: DictConfig) -> Path:
    return Path(config.write_path) / "thresholds.json"


def youden_thresholds(
    config: DictConfig, data: dict, name: str, zs: list[NDArray]
) -> list[float]:
    """Youden thresholds per seed on validation (computed there, stored, reused on test)."""
    path = thresholds_path(config)
    stored = json.loads(path.read_text()) if path.exists() else {}
    if data["split"] == "validation":
        stored[name] = [youden_threshold(data["y"], z)[0] for z in zs]
        path.write_text(json.dumps(stored, indent=1))
    assert name in stored and len(stored[name]) >= len(zs), (
        f"{name}: no validation thresholds"
    )
    return stored[name][: len(zs)]


def evaluate_models(
    config: DictConfig, data: dict, ix: dict, scores: dict
) -> dict[str, pd.DataFrame]:
    rows, cal_rows = [], []
    for name, (label, zs) in scores.items():
        ts = youden_thresholds(config, data, name, zs)
        cals = []
        for seed, (z, t) in enumerate(zip(zs, ts, strict=True)):
            m, cal = seed_metrics(data, ix, z, t)
            rows.append({"model": name, "label": label, "seed": seed, **m})
            cals.append(cal)
        cal_mean = np.mean(cals, axis=0)  # per slide: mean over seeds (Appendix F)
        cal_rows.append(
            {
                "model": name,
                "seeds": len(zs),
                "low_prevalence_slides": len(cal_mean),
                "calibrated_AUPRC_median": float(np.median(cal_mean))
                if len(cal_mean)
                else np.nan,
                "calibrated_AUPRC_q1": float(np.quantile(cal_mean, 0.25))
                if len(cal_mean)
                else np.nan,
                "calibrated_AUPRC_q3": float(np.quantile(cal_mean, 0.75))
                if len(cal_mean)
                else np.nan,
                "calibrated_AUPRC_mean": float(np.mean(cal_mean))
                if len(cal_mean)
                else np.nan,
            }
        )
    per_seed = pd.DataFrame(rows)
    metrics = [c for c in per_seed.columns if c not in ("model", "label", "seed")]
    summary = (
        per_seed.groupby(["model", "label"], sort=False)[metrics].mean().reset_index()
    )
    summary.insert(2, "seeds", per_seed.groupby("model", sort=False).size().to_numpy())
    summary = summary.merge(pd.DataFrame(cal_rows).drop(columns="seeds"), on="model")
    return {"per_seed": per_seed, "summary": summary}


def split_counts(data: dict, ix: dict) -> dict:
    return {
        "split": data["split"],
        "slides": len(data["slides"]),
        "positive_slides": int(ix["positive_slides"].sum()),
        "nuclei": len(data["y"]),
        "carcinoma_nuclei": int((data["y"] == 1).sum()),
        "carcinoma_slide_nuclei_without_label_row": int(data["unmatched"]),
        "foci_per_bin": dict(
            zip(ix["bin_labels"], ix["foci_binned"].values(), strict=True)
        ),
        "carcinoma_nuclei_per_bin": dict(
            zip(ix["bin_labels"], ix["focus_nuclei_binned"].values(), strict=True)
        ),
        "benign_nuclei_per_band": {
            lab: int((ix["band"] == j).sum()) for j, lab in enumerate(ix["band_labels"])
        },
        "low_prevalence_slides": len(ix["low_slides"]),
    }


def write_tables(out: Path, tables: dict[str, pd.DataFrame], counts: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for k, t in tables.items():
        t.to_csv(out / f"{k}.csv", index=False)
    (out / "counts.json").write_text(json.dumps(counts, indent=1))


def stage_evaluate(config: DictConfig) -> None:
    from visualization.small_foci_figures import recall_fpr_figure

    data = load_split(config, config.split)
    ix = eval_index(config, data)
    scores = all_scores(config, data)
    tables = evaluate_models(config, data, ix, scores)
    out = Path(config.write_path) / "results" / config.split
    write_tables(out, tables, split_counts(data, ix))
    recall_fpr_figure(
        tables["per_seed"],
        ix["bin_labels"],
        ix["band_labels"],
        out / f"recall_fpr_{config.split}",
        config.split,
    )
    print(tables["summary"].drop(columns="label").to_string(index=False))


# --------------------------------------------------------------------------
# Step 0: late fusion of the main and the self-only model
# --------------------------------------------------------------------------


def stage_fusion(config: DictConfig) -> None:
    from visualization.small_foci_figures import fusion_figure, recall_fpr_figure

    f = config.fusion
    weights = [float(w) for w in f.weights]
    out = Path(config.write_path) / "results" / "fusion"
    splits = {}
    for split in ("validation", "test"):
        data = load_split(config, split)
        ix = eval_index(config, data)
        main = existing_scores(config, "main", data)
        self_ = existing_scores(config, "self", data)
        splits[split] = (data, ix, main, self_)

    # weight curve on validation (w = weight of the main model)
    data, ix, main, self_ = splits["validation"]
    curve = []
    for w in tqdm(weights, desc="fusion weights"):
        for seed, (zm, zs) in enumerate(zip(main, self_, strict=True)):
            z = w * zm + (1 - w) * zs
            t = youden_threshold(data["y"], z)[0]
            m, _ = seed_metrics(data, ix, z, t)
            curve.append({"w_main": w, "seed": seed, **m})
    curve = pd.DataFrame(curve)
    mean_curve = curve.groupby("w_main").mean(numeric_only=True).reset_index()
    best = float(mean_curve.loc[mean_curve[str(f.selection_metric)].idxmax(), "w_main"])

    results = {}
    for split, (data, ix, main, self_) in splits.items():
        scores = {
            "main": (config.existing_models.main.label, main),
            "self": (config.existing_models.self.label, self_),
            "fusion_mean": (
                "Fusion: mean logit (w = 0.5)",
                [0.5 * a + 0.5 * b for a, b in zip(main, self_, strict=True)],
            ),
            "fusion_w": (
                f"Fusion: w = {best:g} (validation {f.selection_metric})",
                [best * a + (1 - best) * b for a, b in zip(main, self_, strict=True)],
            ),
        }
        tables = evaluate_models(config, data, ix, scores)
        write_tables(out / split, tables, split_counts(data, ix))
        recall_fpr_figure(
            tables["per_seed"],
            ix["bin_labels"],
            ix["band_labels"],
            out / split / f"recall_fpr_fusion_{split}",
            f"Step 0, {split}",
        )
        results[split] = tables
        print(f"== fusion, {split}")
        print(tables["summary"].drop(columns="label").to_string(index=False))
    curve.to_csv(out / "weight_curve_validation.csv", index=False)
    (out / "selected_weight.json").write_text(
        json.dumps({"w_main": best, "metric": str(f.selection_metric)})
    )
    fusion_figure(mean_curve, best, out / "weight_curve_validation")


# --------------------------------------------------------------------------
# Predictions of the study's training runs (own commit and config)
# --------------------------------------------------------------------------


def training_info(client: MlflowClient, run_id: str, tmp: Path) -> dict:
    run = client.get_run(run_id)
    commit = run.data.tags.get("mlflow.source.git.commit")
    assert commit, f"{run_id}: no commit tag"
    (tmp / run_id).mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(
        Path(
            client.download_artifacts(
                run_id, "configs/config-resolved.yaml", str(tmp / run_id)
            )
        ).read_text()
    )
    files = [f.path for f in client.list_artifacts(run_id, "checkpoints")]
    assert len(files) == 1, f"{run_id}: {len(files)} checkpoints"
    return {
        "training_run": run_id,
        "commit": commit,
        "checkpoint": files[0],
        "checkpoint_epoch": int(CKPT_EPOCH.search(files[0]).group(1)),
        "train_cfg": cfg,
        "status": run.info.status,
        "hours": (run.info.end_time - run.info.start_time) / 3.6e6
        if run.info.end_time
        else None,
    }


def stage_predict(config: DictConfig) -> None:
    client = MlflowClient()
    tmp = Path(config.tmp_dir)
    tmp.mkdir(parents=True, exist_ok=True)
    split = config.split
    slides = split_slides(config, split)
    s = config.validation_split
    pending = defaultdict(list)
    for m in config.study_models.values():
        for run in m.training_runs:
            output = Path(config.predictions_dir) / f"{run}_{split}.parquet"
            if output.exists():
                continue
            info = training_info(client, run, tmp)
            d = info["train_cfg"]["data"]
            assert (d["split_size"], d["split_stratify_col"], d["split_group_col"]) == (
                float(s.size),
                s.stratify_col,
                s.group_col,
            ), f"{run}: another validation split"
            assert info["status"] == "FINISHED", f"{run}: {info['status']}"
            pending[info["commit"]].append((info, output))
    for commit, items in pending.items():
        runs = []
        for info, _ in items:
            t = info["training_run"]
            local = Path(client.download_artifacts(t, info["checkpoint"], str(tmp / t)))
            if local.is_dir():
                (local,) = list(local.rglob("*.ckpt"))
            cfg_path = tmp / t / "config-remapped.yaml"
            cfg_path.write_text(yaml.safe_dump(remap(info["train_cfg"], config)))
            runs.append({"checkpoint": str(local), "config": str(cfg_path)})
        # one inference job per model configuration (the collator's k must be shared)
        groups = defaultdict(list)
        for (info, output), r in zip(items, runs, strict=True):
            groups[int(info["train_cfg"]["data"]["k"])].append((r, output))
        for k, group in groups.items():
            predict_at_commit(
                config,
                commit,
                slides,
                [r for r, _ in group],
                [o for _, o in group],
                f"{split}_k{k}",
            )


def predict_at_commit(
    config: DictConfig,
    commit: str,
    slides: pd.DataFrame,
    runs: list[dict],
    outputs: list[Path],
    tag: str,
) -> None:
    tmp = Path(config.tmp_dir)
    slides_path = tmp / f"{commit[:8]}_{tag}_slides.parquet"
    slides.to_parquet(slides_path, index=False)
    job = tmp / f"{commit[:8]}_{tag}_job.json"
    job.write_text(
        json.dumps(
            {
                "slides": str(slides_path),
                "runs": runs,
                "outputs": [str(o) for o in outputs],
                "device": str(config.inference.device),
                "logits": True,
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


# --------------------------------------------------------------------------
# Step 3: test, paired bootstrap over the canonical slide resamples
# --------------------------------------------------------------------------


def weighted_median(values: NDArray, weights: NDArray) -> NDArray:
    """Median of the multiset with integer `weights` (resamples, n); `values` (n,)."""
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[:, order]
    cum = np.cumsum(w, axis=1)
    total = cum[:, -1]
    lo = np.array(
        [
            np.searchsorted(c, (t - 1) // 2, side="right")
            for c, t in zip(cum, total, strict=True)
        ]
    )
    hi = np.array(
        [
            np.searchsorted(c, t // 2, side="right")
            for c, t in zip(cum, total, strict=True)
        ]
    )
    return (v[lo] + v[hi]) / 2


def replicates(
    config: DictConfig,
    data: dict,
    ix: dict,
    zs: list[NDArray],
    ts: list[float],
    weights: NDArray,
) -> dict[str, NDArray]:
    """Seed-averaged bootstrap replicates of the shared metrics (resamples,) each."""
    b = config.bootstrap
    y, slide = data["y"], data["slide"]
    n_slides = len(data["slides"])
    lo, hi = -float(config.fusion.logit_clip) - 1, float(config.fusion.logit_clip) + 1
    lo = min(lo, min(float(z.min()) for z in zs))
    hi = max(hi, max(float(z.max()) for z in zs))
    out: dict[str, list[NDArray]] = defaultdict(list)
    cal_per_slide = []
    for z, t in zip(zs, ts, strict=True):
        exact = {"AUPRC": average_precision_score(y, z), "AUROC": roc_auc_score(y, z)}
        n_bins = int(b.histogram.n_bins)
        while True:
            bins = bin_logits(z, lo, hi, n_bins)
            flat = slide.astype(np.int64) * n_bins + bins
            pos = (
                np.bincount(flat[y == 1], minlength=n_slides * n_bins)
                .reshape(n_slides, n_bins)
                .astype(np.float32)
            )
            neg = (
                np.bincount(flat[y == 0], minlength=n_slides * n_bins)
                .reshape(n_slides, n_bins)
                .astype(np.float32)
            )
            full = hist_metrics(
                pos.sum(0, dtype=np.float64), neg.sum(0, dtype=np.float64)
            )
            err = max(abs(float(full[k]) - exact[k]) for k in exact)
            if err <= float(b.histogram.tolerance) or n_bins >= int(
                b.histogram.max_bins
            ):
                break
            n_bins *= 2
        assert err <= float(b.histogram.tolerance), (
            f"histogram grid too coarse ({err:.1e})"
        )
        for start in range(0, len(weights), 100):
            w = weights[start : start + 100].astype(np.float32)
            for k, v in hist_metrics(
                (w @ pos).astype(np.float64), (w @ neg).astype(np.float64)
            ).items():
                out[f"{k}#{start}"].append(v)
        del pos, neg
        hit = z >= t
        counts = {}
        for b_, lab in enumerate(ix["bin_labels"]):
            counts[f"recall_bin_{lab}"] = ix["bin"] == b_
        counts["recall_small"] = np.isin(ix["bin"], ix["small_bins"])
        for j, lab in enumerate(ix["band_labels"]):
            counts[f"FPR_out_{lab}um"] = ix["band"] == j
        counts["recall"], counts["FPR"] = y == 1, y == 0
        for k, sel in counts.items():
            num = np.bincount(slide[sel & hit], minlength=n_slides).astype(np.float64)
            den = np.bincount(slide[sel], minlength=n_slides).astype(np.float64)
            with np.errstate(invalid="ignore"):
                out[k].append((weights @ num) / (weights @ den))
        _, cal = seed_metrics_low(data, ix, z)
        cal_per_slide.append(cal)
    res = {}
    for k in ("AUPRC", "AUROC"):
        parts = sorted(
            (key for key in out if key.startswith(f"{k}#")),
            key=lambda s: int(s.split("#")[1]),
        )
        res[k] = np.mean(
            [np.concatenate([out[p][i] for p in parts]) for i in range(len(zs))], axis=0
        )
    for k, v in out.items():
        if "#" not in k:
            res[k] = np.mean(v, axis=0)
    low = ix["low_slides"]
    res["calibrated_AUPRC_median"] = weighted_median(
        np.mean(cal_per_slide, axis=0), weights[:, low]
    )
    return res


def seed_metrics_low(data: dict, ix: dict, z: NDArray) -> tuple[None, NDArray]:
    y, st = data["y"], ix["starts"]
    return None, np.array(
        [
            calibrated_ap(
                y[st[s] : st[s + 1]], z[st[s] : st[s + 1]], ix["low_c"][int(s)]
            )
            for s in ix["low_slides"]
        ]
    )


def stage_test(config: DictConfig) -> None:
    from visualization.small_foci_figures import recall_fpr_figure

    b = config.bootstrap
    data = load_split(config, "test")
    ix = eval_index(config, data)
    scores = all_scores(config, data)
    keep = ["main", str(b.reference), *[str(m) for m in b.models]]
    missing = [m for m in keep if m not in scores]
    assert not missing, f"no test predictions of {missing}"
    scores = {m: scores[m] for m in keep}
    tables = evaluate_models(config, data, ix, scores)
    out = Path(config.write_path) / "results" / "test"
    write_tables(out, tables, split_counts(data, ix))
    recall_fpr_figure(
        tables["per_seed"],
        ix["bin_labels"],
        ix["band_labels"],
        out / "recall_fpr_test",
        "test",
    )

    r = np.load(b.resamples)
    assert int(r["seed"]) == int(b.resamples_seed)
    assert list(r["slide_ids"]) == data["slides"]["slide_id"].tolist(), (
        "resamples are for other slides"
    )
    assert np.array_equal(r["positive"], ix["positive_slides"]), (
        "resample strata differ"
    )
    weights = r["multiplicities"].astype(np.float64)
    reps = {}
    for name, (_, zs) in scores.items():
        ts = youden_thresholds(config, data, name, zs)
        reps[name] = replicates(config, data, ix, zs, ts, weights)
    q = (1 - float(b.ci_level)) / 2 * 100
    summary = tables["summary"].set_index("model")
    rows = []
    pairs = [(m, str(b.reference)) for m in b.models] + [
        (m, "main") for m in [str(b.reference), *b.models]
    ]
    for a, ref in pairs:
        for k in reps[a]:
            diff = reps[a][k] - reps[ref][k]
            point = (
                float(summary.loc[a, k] - summary.loc[ref, k])
                if k in summary.columns
                else float(np.nan)
            )
            rows.append(
                {
                    "model": a,
                    "reference": ref,
                    "metric": k,
                    "difference": point,
                    "ci_lo": float(np.nanpercentile(diff, q)),
                    "ci_hi": float(np.nanpercentile(diff, 100 - q)),
                    "p": bootstrap_p(diff[np.isfinite(diff)]),
                }
            )
    comp = pd.DataFrame(rows)
    is_auprc = comp["metric"] == "AUPRC"
    comp["p_holm"] = np.nan
    comp.loc[is_auprc, "p_holm"] = holm(comp.loc[is_auprc, "p"].to_numpy())
    comp.to_csv(out / "bootstrap_comparisons.csv", index=False)
    ci = []
    for name, rep in reps.items():
        for k, v in rep.items():
            ci.append(
                {
                    "model": name,
                    "metric": k,
                    "ci_lo": float(np.nanpercentile(v, q)),
                    "ci_hi": float(np.nanpercentile(v, 100 - q)),
                }
            )
    pd.DataFrame(ci).to_csv(out / "bootstrap_cis.csv", index=False)
    print(
        comp[
            comp["metric"].isin(["AUPRC", "recall_small", "calibrated_AUPRC_median"])
        ].to_string(index=False)
    )


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


SHOW = ("AUPRC", "AUROC", "recall_small")


def fmt_table(df: pd.DataFrame, cols: list[str], digits: int = 3) -> str:
    head = "| " + " | ".join(cols) + " |\n| " + " | ".join("---" for _ in cols) + " |\n"
    lines = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            cells.append(
                f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)
            )
        lines.append("| " + " | ".join(cells) + " |")
    return head + "\n".join(lines) + "\n"


def selection(
    config: DictConfig, summary: pd.DataFrame, bins: list[str]
) -> pd.DataFrame:
    """Step 2 rule on validation: recall up in both smallest bins, AUPRC down by at most max_auprc_drop."""
    s = summary.set_index("model")
    ref = s.loc["control"]
    small = [f"recall_bin_{b}" for b in bins[: int(config.evaluation.small_bins)]]
    rows = []
    for name in config.study_models:
        if name == "control" or name not in s.index:
            continue
        r = s.loc[name]
        d = {c: float(r[c] - ref[c]) for c in [*small, "recall_small", "AUPRC"]}
        ok = all(d[c] > 0 for c in small) and d["AUPRC"] >= -float(
            config.selection.max_auprc_drop
        )
        rows.append(
            {
                "model": name,
                "seeds": int(r["seeds"]),
                **{f"delta_{k}": v for k, v in d.items()},
                "eligible": ok,
            }
        )
    t = pd.DataFrame(rows)
    if t.empty:
        return t
    t = t.sort_values(
        ["eligible", "delta_recall_small"], ascending=[False, False], ignore_index=True
    )
    t["selected"] = t["eligible"] & (t.index < int(config.selection.max_selected))
    return t


def run_table(config: DictConfig, client: MlflowClient) -> pd.DataFrame:
    rows = []
    for name, m in config.study_models.items():
        for seed, run_id in enumerate(m.training_runs):
            run = client.get_run(run_id)
            i, met = run.info, run.data.metrics
            rows.append(
                {
                    "model": name,
                    "seed": seed,
                    "training_run": run_id,
                    "status": i.status,
                    "commit": run.data.tags.get("mlflow.source.git.commit", "")[:8],
                    "hours": (i.end_time - i.start_time) / 3.6e6
                    if i.end_time
                    else np.nan,
                    "best_epoch": met.get("best/epoch", np.nan),
                    "best_validation_loss": met.get("best/validation/loss", np.nan),
                }
            )
    return pd.DataFrame(rows)


def stage_report(config: DictConfig) -> None:
    client = MlflowClient()
    res = Path(config.write_path) / "results"
    res.mkdir(parents=True, exist_ok=True)
    parts = [
        "# small_foci: interventions against context dilution on small foci (PANDA Radboud)\n"
    ]
    parts.append(
        "Shared evaluation: foci = components of the G3-5 mask after a 50 µm closing (7287d498); "
        "recall at each seed's Youden threshold (validation nuclei) pooled over the carcinoma nuclei of "
        "the foci (≥ 5 carcinoma nuclei) in each closed-area bin; FPR of benign nuclei of positive slides "
        "by distance outside the nearest focus; calibrated AUPRC (π₀ = 0.385) median over positive slides "
        "with < 5 % carcinoma and ≥ 50 carcinoma nuclei (per slide, the seed mean). Metrics per seed, then the "
        "mean. µm from the slide table mpp 0.4862 µm/px (= Radboud TIFF header).\n"
    )
    runs = run_table(config, client)
    if len(runs):
        runs.to_csv(res / "training_runs.csv", index=False)
        parts.append(
            "## Training runs\n\n"
            + fmt_table(
                runs,
                [
                    "model",
                    "seed",
                    "training_run",
                    "status",
                    "commit",
                    "hours",
                    "best_epoch",
                ],
                1,
            )
        )
    for sub, title in (
        ("fusion/validation", "Step 0 (fusion), validation"),
        ("fusion/test", "Step 0 (fusion), test"),
        ("validation", "Validation"),
        ("test", "Test (step 3)"),
    ):
        path = res / sub / "summary.csv"
        if not path.exists():
            continue
        summ = pd.read_csv(path)
        counts = json.loads((res / sub / "counts.json").read_text())
        bins = list(counts["foci_per_bin"])
        bands = list(counts["benign_nuclei_per_band"])
        cols = [
            "model",
            "seeds",
            *SHOW,
            *[f"recall_bin_{b}" for b in bins[:2]],
            *[f"FPR_out_{b}um" for b in bands],
            "calibrated_AUPRC_median",
            "low_prevalence_slides",
        ]
        parts.append(f"## {title}\n\n" + fmt_table(summ, cols))
        parts.append(
            f"\n{counts['slides']} slides ({counts['positive_slides']} positive), {counts['nuclei']:,} nuclei; "
            f"foci in the two smallest bins: {list(counts['foci_per_bin'].values())[:2]}; "
            f"benign nuclei per band: {list(counts['benign_nuclei_per_band'].values())}.\n"
        )
        if sub == "validation":
            sel = selection(config, summ, bins)
            if len(sel):
                sel.to_csv(res / "selection_validation.csv", index=False)
                parts.append(
                    "\n### Selection (step 2 rule)\n\n"
                    + fmt_table(sel, list(sel.columns), 4)
                )
    comp = res / "test" / "bootstrap_comparisons.csv"
    if comp.exists():
        c = pd.read_csv(comp)
        c = c[
            c["metric"].isin(
                [
                    "AUPRC",
                    "AUROC",
                    "recall_small",
                    "FPR_out_0-50um",
                    "calibrated_AUPRC_median",
                ]
            )
        ]
        parts.append(
            "## Step 3 paired bootstrap (test)\n\n"
            + fmt_table(
                c,
                [
                    "model",
                    "reference",
                    "metric",
                    "difference",
                    "ci_lo",
                    "ci_hi",
                    "p",
                    "p_holm",
                ],
                4,
            )
        )
    (res / "report.md").write_text("\n".join(parts))
    preflight(res, set(runs["training_run"]) if len(runs) else set())
    if config.log_mlflow:
        log_results(config, res)


def preflight(out: Path, allowed: set[str]) -> None:
    """No credentials and no 32-hex token except run ids in the text outputs (no slide ids)."""
    for path in out.rglob("*"):
        if path.suffix in {".csv", ".json", ".md"}:
            text = path.read_text()
            assert not CREDENTIAL_URL.search(text), path
            bad = set(HEX32.findall(text)) - allowed
            assert not bad, f"{path}: unexpected 32-hex tokens"


def log_results(config: DictConfig, res: Path) -> None:
    info = git_info(*CODE)
    with mlflow.start_run(
        experiment_id=str(config.metadata.experiment_id),
        run_name="small_foci: shared evaluation",
        tags={"study": "small_foci", "role": "evaluation", **info},
    ) as run:
        mlflow.log_artifacts(str(res), "results")
        print(f"MLflow run {run.info.run_id}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


@with_cli_args(["+postprocessing=small_foci_radboud"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
def main(config: DictConfig) -> None:
    OmegaConf.resolve(config)
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    for stage in config.stages:
        print(f"== {stage}")
        STAGES[stage](config)


STAGES = {
    "geometry": stage_geometry,
    "foci_table": stage_foci_table,
    "predict": stage_predict,
    "evaluate": stage_evaluate,
    "fusion": stage_fusion,
    "test": stage_test,
    "report": stage_report,
}


if __name__ == "__main__":
    main()
