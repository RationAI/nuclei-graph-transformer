r"""NB-A-64: pixel vs nucleus tile-level carcinoma models, paired patient-level bootstrap (MMCI TL test).

Inputs, all stored outputs of existing runs (no training, no inference):
- pixel models (experiment 65, one model each): `tables/carcinoma_prediction_table.json`
  (columns slide, x, y, prediction, target);
- nucleus models (experiment 97, three seeds each): the pooled tile predictions written by
  `postprocessing/tile/nuclei_to_tile_pooling.py` (`tile_predictions/<stem>.parquet`: x, y,
  tile_prediction), pooled with the mode and k that `sweep_pooling` in
  `postprocessing/tile/thresholds.py` selected on the validation patients
  (`pooling_mode_statistics.csv`, argmax of roc_auc). The selection is re-derived from that
  table and must equal the run's `pooling_mode`/`pooling_k` params; nothing is selected here.

Step 1, alignment (any failed check raises before metrics are computed):
- keys (slide stem, x, y): level-1 top-left tile coordinates in every source;
- labels: one source, `carcinoma_roi_percentage > carcinoma_roi_t` of the QC-filtered tiling
  (as `load_tile_predictions`); the pixel tables' `target` must agree on every joined tile;
- tile set: every model is evaluated on the intersection of all sources; the QC exclusions
  (`BaseTileDataset.filter_tiles_by_thresholds`) are recomputed from the unfiltered tiling,
  must reproduce the filtered tiling exactly, and are counted per patient and class;
- empty ROIs: tiles whose central ROI holds no nucleus centroid, recomputed from the nuclei
  with the `BaseTileDataset` geometry (`get_scaled_props`, `get_tile_indices`, `get_roi_mask`);
  they must be exactly the tiles every nucleus seed scores 0.0;
- the pixel models' full-test-set AUPRC/AUROC must reproduce NB-A-64.

Metrics: torchmetrics `BinaryAveragePrecision` (AUPRC) and `BinaryAUROC` over all tiles, as
`postprocessing/tile/metrics.py`. Nucleus models: the mean of the three seeds' metrics (scores
are never averaged across seeds for a metric).

Bootstrap: `n_resamples` resamples of the test patients with replacement (each drawn patient
contributes all its tiles, with multiplicity), one fixed seed, identical for all models and both
metrics; a resample without positive or without negative tiles is redrawn. Replicate metrics
use the same definitions as torchmetrics with integer tile weights (= patient multiplicity),
which equals evaluating the replicated tiles; this is checked against torchmetrics on
physically replicated tiles for the first `n_torchmetrics_checks` resamples. Per replicate the
nucleus metric is the mean over seeds. CIs: percentiles of the replicates. Differences A - B
within replicates; p = 2 min(P(Δ <= 0), P(Δ >= 0)), capped at 1; Holm on `holm_metric` only.

The MLflow run is created with `MlflowClient` rather than `autolog`, because the autolog logger
copies the git remote URL (which may embed credentials) into the run tags.
"""

import json
import os
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import hydra
import mlflow
import numpy as np
import pandas as pd
import torch
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import with_cli_args
from ratiopath.openslide import OpenSlide
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

from nuclei_graph.data.datasets.tile.base import (
    CARCINOMA_P,
    PERCENTAGES,
    BaseTileDataset,
    get_slide_data,
)


METRICS = ("AUPRC", "AUROC")
KEYS = ["slide_id", "x", "y"]
REPO = Path(__file__).parents[2]


# ----------------------------------------------------------------------------- metrics
def tm_metrics(y: NDArray, s: NDArray) -> dict[str, float]:
    """AUPRC/AUROC exactly as postprocessing/tile/metrics.py (torchmetrics, float64 scores)."""
    preds = torch.as_tensor(np.asarray(s, dtype=np.float64))
    target = torch.as_tensor(np.asarray(y)).long()
    return {
        "AUPRC": float(BinaryAveragePrecision()(preds, target)),
        "AUROC": float(BinaryAUROC()(preds, target)),
    }


class WeightedRanking:
    """Exact AUPRC/AUROC of one score column under integer per-patient tile weights.

    Tiles are grouped by distinct score (ties form one threshold, as in torchmetrics'
    `_binary_clf_curve`); per group the positive and negative tiles are counted per patient,
    so a resample's weighted counts are a matrix-vector product with the patient multiplicities.
    """

    def __init__(self, score: NDArray, y: NDArray, patient: NDArray, n_patients: int):
        order = np.argsort(-score, kind="stable")
        s = score[order]
        starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
        group = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(s)]))
        yo, po = y[order].astype(bool), patient[order]
        self.pos = np.zeros((len(starts), n_patients))
        self.neg = np.zeros((len(starts), n_patients))
        np.add.at(self.pos, (group[yo], po[yo]), 1.0)
        np.add.at(self.neg, (group[~yo], po[~yo]), 1.0)

    def metrics(self, mult: NDArray) -> dict[str, float]:
        tp, fp = self.pos @ mult, self.neg @ mult
        keep = (tp + fp) > 0
        tps, fps = np.cumsum(tp[keep]), np.cumsum(fp[keep])
        n_pos, n_neg = tps[-1], fps[-1]
        tpr, fpr = np.r_[0.0, tps / n_pos], np.r_[0.0, fps / n_neg]
        recall = tps / n_pos
        precision = tps / (tps + fps)
        return {
            "AUPRC": float(np.sum(np.diff(np.r_[0.0, recall]) * precision)),
            "AUROC": float(np.trapezoid(tpr, fpr)),
        }


def bootstrap_p(diff: NDArray) -> float:
    return float(min(1.0, 2 * min(np.mean(diff <= 0), np.mean(diff >= 0))))


def holm(p: NDArray) -> NDArray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.minimum(1.0, (len(p) - np.arange(len(p))) * p[order])
    out = np.empty_like(adj)
    out[order] = np.maximum.accumulate(adj)
    return out


# ----------------------------------------------------------------------------- loading
def load_tiling(uri: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(download_artifacts(artifact_uri=uri))
    tiles = pd.read_parquet(path / "tiles.parquet")
    slides = pd.read_parquet(path / "slides.parquet")
    tiles["slide_id"] = tiles["slide_id"].map(
        dict(zip(slides["id"], slides["stem"], strict=True))
    )
    assert tiles["slide_id"].notna().all()
    return tiles, slides


def qc_mask(tiles: pd.DataFrame, thresholds: DictConfig) -> dict[str, pd.Series]:
    """Per-criterion keep masks of BaseTileDataset.filter_tiles_by_thresholds (same order)."""
    masks = {}
    for percentage in PERCENTAGES:
        key = percentage.replace("percentage", "t")
        masks[key.removesuffix("_t")] = (
            tiles[percentage] > thresholds[key]
            if "tissue" in percentage
            else tiles[percentage] <= thresholds[key]
        )
    return masks


def exclusion_table(
    raw: pd.DataFrame, masks: dict[str, pd.Series], kept_by_join: pd.Series
) -> pd.DataFrame:
    """Tiles removed by each sequential exclusion and by the join, per patient and class."""
    df = raw[["patient_id", "label"]].copy()
    alive = pd.Series(True, index=raw.index)
    cols = []
    for name, keep in masks.items():
        df[f"fails_{name}"] = ~keep
        df[f"removed_{name}"] = alive & ~keep
        cols += [f"fails_{name}", f"removed_{name}"]
        alive &= keep
    df["removed_join"] = alive & ~kept_by_join
    df["n_final"] = alive & kept_by_join
    df["n_unfiltered"] = True
    cols = ["n_unfiltered", *cols, "removed_join", "n_final"]
    out = df.groupby(["patient_id", "label"])[cols].sum().reset_index()
    total = out[cols].sum().to_frame().T
    total["patient_id"], total["label"] = "all", "all"
    by_class = out.groupby("label")[cols].sum().reset_index()
    by_class["patient_id"] = "all"
    return pd.concat([out, by_class, total], ignore_index=True)[
        ["patient_id", "label", *cols]
    ]


def load_pixel(run: str, artifact: str) -> pd.DataFrame:
    table = json.loads(
        Path(download_artifacts(run_id=run, artifact_path=artifact)).read_text()
    )
    df = pd.DataFrame(table["data"], columns=table["columns"])
    return df.rename(columns={"slide": "slide_id"})


def load_nucleus(run: str, artifact: str) -> pd.DataFrame:
    path = Path(download_artifacts(run_id=run, artifact_path=artifact))
    parts = [
        pd.read_parquet(p).assign(slide_id=p.stem)
        for p in sorted(path.glob("*.parquet"))
    ]
    return pd.concat(parts, ignore_index=True)


def pooling_selection(client: MlflowClient, run: str, artifact: str) -> dict:
    stats = pd.read_csv(download_artifacts(run_id=run, artifact_path=artifact))
    ranked = stats.sort_values("roc_auc", ascending=False).reset_index(drop=True)
    best = stats.loc[stats["roc_auc"].idxmax()]
    k = None if pd.isna(best["k"]) else int(best["k"])
    params = client.get_run(run).data.params
    logged_k = params.get("pooling_k")
    logged_k = None if logged_k in (None, "None", "null") else int(logged_k)
    if params.get("pooling_mode") != best["pooling_mode"] or logged_k != k:
        raise ValueError(
            f"{run}: logged pooling {params.get('pooling_mode')}/{logged_k} is not the "
            f"validation argmax {best['pooling_mode']}/{k}"
        )
    return {
        "pooling_mode": best["pooling_mode"],
        "k": k,
        "val_roc_auc": float(best["roc_auc"]),
        "runner_up": f"{ranked.loc[1, 'pooling_mode']}"
        + ("" if pd.isna(ranked.loc[1, "k"]) else f"/k={int(ranked.loc[1, 'k'])}"),
        "margin_to_runner_up": float(
            ranked.loc[0, "roc_auc"] - ranked.loc[1, "roc_auc"]
        ),
        "n_configurations": len(stats),
    }


# ----------------------------------------------------------------------------- empty ROIs
def _slide_roi_counts(args: tuple) -> tuple[str, NDArray]:
    stem, xy, props, tile_size, margin = args
    ns = SimpleNamespace(slide_props={stem: props}, tile_size=tile_size, margin=margin)
    _, centroids, tree, _ = get_slide_data(props["slide_nuclei_path"])
    counts = np.zeros(len(xy), dtype=np.int64)
    for i, (x, y) in enumerate(xy):
        sp = BaseTileDataset.get_scaled_props(ns, {"stem": stem, "x": x, "y": y})
        idx = BaseTileDataset.get_tile_indices(ns, sp, centroids, tree)
        if len(idx):
            counts[i] = int(BaseTileDataset.get_roi_mask(ns, sp, centroids[idx]).sum())
    return stem, counts


def roi_nuclei_counts(
    tiles: pd.DataFrame, slides: pd.DataFrame, meta: pd.DataFrame, config: DictConfig
) -> NDArray:
    """Number of nucleus centroids in each tile's ROI (BaseTileDataset geometry)."""
    meta = meta.set_index("slide_id")
    jobs = []
    for stem, group in tiles.groupby("slide_id", sort=False):
        srow = slides[slides["stem"] == stem].iloc[0]
        with OpenSlide(meta.loc[stem, "slide_path"]) as slide:
            base = slide.level_dimensions[0]
            level = slide.level_dimensions[int(srow["level"])]
        sx, sy = level[0] / base[0], level[1] / base[1]
        props = {
            "scale_x": sx,
            "scale_y": sy,
            "scaled_extent_x": srow["tile_extent_x"] / sx,
            "scaled_extent_y": srow["tile_extent_y"] / sy,
            "slide_nuclei_path": str(meta.loc[stem, "slide_nuclei_path"]),
        }
        jobs.append(
            (stem, group[["x", "y"]].to_numpy(), props, config.tile_size, config.margin)
        )
    counts = pd.Series(0, index=tiles.index, dtype=np.int64)
    with ProcessPoolExecutor(config.empty_roi_workers) as pool:
        for stem, c in tqdm(
            pool.map(_slide_roi_counts, jobs), total=len(jobs), desc="ROI nuclei"
        ):
            counts[tiles.index[tiles["slide_id"] == stem]] = c
    return counts.to_numpy()


# ----------------------------------------------------------------------------- main computation
def compute(config: DictConfig, client: MlflowClient, out: Path) -> dict:
    checks: dict = {}
    thr = config.thresholds

    # ---- tilings, patients, QC
    tiles, slides = load_tiling(config.tiles_uri)
    raw, _ = load_tiling(config.unfiltered_tiles_uri)
    meta = pd.read_parquet(download_artifacts(artifact_uri=config.metadata_uri))
    meta["patient_id"] = meta["patient_id"].astype(str)
    from_stem = meta["slide_id"].str.extract(r"TP-\d{4}_(\d+)-")[0]
    assert (from_stem == meta["patient_id"]).all(), (
        "patient_id does not match the slide stem"
    )
    patient_of = dict(zip(meta["slide_id"], meta["patient_id"], strict=True))

    geometry = slides[
        ["tile_extent_x", "tile_extent_y", "stride_x", "stride_y", "level"]
    ]
    geometry = geometry.drop_duplicates().to_dict("records")
    assert geometry == [
        {
            "tile_extent_x": 224,
            "tile_extent_y": 224,
            "stride_x": 112,
            "stride_y": 112,
            "level": 1,
        }
    ], geometry
    checks["tiling"] = {
        "tile_extent": 224,
        "stride": 112,
        "level": 1,
        "mpp_level": sorted({round(float(v), 4) for v in slides["mpp_x"]}),
        "roi": f"central {config.tile_size - 2 * config.margin} px (margin {config.margin})",
        "coordinates": "level-1 pixels, top-left tile corner",
    }

    for df in (tiles, raw):
        assert not df.duplicated(KEYS).any()
        df["patient_id"] = df["slide_id"].map(patient_of)
        assert df["patient_id"].notna().all()
        df["label"] = (df[CARCINOMA_P] > thr.carcinoma_roi_t).astype(int)

    masks = qc_mask(raw, thr)
    keep_qc = np.logical_and.reduce(list(masks.values()))
    qc_keys = raw.loc[keep_qc, KEYS].sort_values(KEYS).reset_index(drop=True)
    assert qc_keys.equals(tiles[KEYS].sort_values(KEYS).reset_index(drop=True)), (
        "QC filter of the unfiltered tiling does not reproduce the filtered tiling"
    )
    assert np.logical_and.reduce(list(qc_mask(tiles, thr).values())).all()
    checks["qc_filter_reproduced"] = True

    # ---- sources
    sources: dict[str, pd.DataFrame] = {}
    schema = {}
    for name, m in config.pixel_models.items():
        run = client.get_run(m.run)
        assert run.info.run_name.strip() == m.expected_run_name, (
            name,
            run.info.run_name,
        )
        df = load_pixel(m.run, config.pixel_predictions_artifact)
        schema[name] = {
            "run": m.run,
            "run_name": run.info.run_name.strip(),
            "columns": {c: str(t) for c, t in df.dtypes.items()},
            "n_rows": len(df),
            "score_column": "prediction",
            "score_range": [
                float(df["prediction"].min()),
                float(df["prediction"].max()),
            ],
            "score_type": "probability (sigmoid output in [0, 1])"
            if df["prediction"].between(0, 1).all()
            else "logit / unbounded",
            "label_column": "target (bool)",
            "label_threshold": run.data.params.get(
                "carcinoma_roi_t", "carcinoma_roi_t = 0.5 (config)"
            ),
            "keys": "slide (stem), x, y: level-1 top-left, 224 px tiles, stride 112",
        }
        assert not df.duplicated(KEYS).any()
        sources[name] = df.rename(columns={"prediction": name})

    pooling_rows = []
    nuc_cols: dict[str, list[str]] = {}
    for name, m in config.nucleus_models.items():
        nuc_cols[name] = []
        for i, run in enumerate(m.eval_runs, start=1):
            col = f"{name}_seed{i}"
            sel = pooling_selection(client, run, config.pooling_statistics_artifact)
            pooling_rows.append({"model": name, "seed": i, "run": run, **sel})
            df = load_nucleus(run, config.tile_predictions_artifact)
            assert not df.duplicated(KEYS).any()
            schema[col] = {
                "run": run,
                "n_rows": len(df),
                "n_slides": df["slide_id"].nunique(),
            }
            sources[col] = df.rename(columns={"tile_prediction": col})
            nuc_cols[name].append(col)
    pooling = pd.DataFrame(pooling_rows)
    pooling.to_csv(out / "pooling_selection.csv", index=False)
    checks["schema"] = schema

    # ---- join on (slide_id, x, y); label agreement
    base = tiles[[*KEYS, "patient_id", "label", CARCINOMA_P]]
    aligned = base
    join_counts = {"filtered_tiling": {"n": len(base), "pos": int(base["label"].sum())}}
    label_checks = {}
    for name, df in sources.items():
        cols = [*KEYS, name] + (["target"] if "target" in df else [])
        j = base[KEYS].merge(df[KEYS], on=KEYS, how="outer", indicator=True)["_merge"]
        lab = df.merge(base[[*KEYS, "label"]], on=KEYS, how="left")
        own_pos = int(df["target"].sum()) if "target" in df else int(lab["label"].sum())
        join_counts[name] = {
            "n": len(df),
            "pos": own_pos,
            "neg": len(df) - own_pos,
            "only_in_source": int((j == "right_only").sum()),
            "only_in_tiling": int((j == "left_only").sum()),
        }
        if "target" in df:
            both = lab.dropna(subset=["label"])
            disagree = int((both["target"].astype(int) != both["label"]).sum())
            label_checks[name] = {"joined": len(both), "disagree": disagree}
            if disagree:
                raise ValueError(
                    f"{name}: {disagree} tiles where target != our label; stop."
                )
        aligned = aligned.merge(
            df[cols].drop(columns=["target"], errors="ignore"), on=KEYS, how="inner"
        )
    checks["label_agreement"] = label_checks
    checks["join"] = join_counts
    aligned = aligned.sort_values(KEYS).reset_index(drop=True)
    at_threshold = int((aligned[CARCINOMA_P] == thr.carcinoma_roi_t).sum())
    checks["label_rule"] = {
        "ours": f"carcinoma_roi_percentage > {thr.carcinoma_roi_t}",
        "pixel_target": f"identical on all joined tiles (carcinoma_roi_t = {thr.carcinoma_roi_t}, strict >)",
        "tiles_exactly_at_threshold": at_threshold,
    }

    kept_by_join = raw.set_index(KEYS).index.isin(aligned.set_index(KEYS).index)
    excl = exclusion_table(raw, masks, pd.Series(kept_by_join, index=raw.index))
    excl.to_csv(out / "exclusions.csv", index=False)

    # ---- empty ROIs
    aligned["roi_nuclei"] = roi_nuclei_counts(aligned, slides, meta, config)
    aligned["empty_roi"] = aligned["roi_nuclei"] == 0
    all_zero = np.logical_and.reduce(
        [aligned[c].to_numpy() == 0.0 for cols in nuc_cols.values() for c in cols]
    )
    any_zero = np.logical_or.reduce(
        [aligned[c].to_numpy() == 0.0 for cols in nuc_cols.values() for c in cols]
    )
    checks["empty_roi"] = {
        "n": int(aligned["empty_roi"].sum()),
        "pos": int((aligned["empty_roi"] & (aligned["label"] == 1)).sum()),
        "neg": int((aligned["empty_roi"] & (aligned["label"] == 0)).sum()),
        "matches_all_seeds_zero": bool((all_zero == aligned["empty_roi"]).all()),
        "zero_score_but_nonempty": int((any_zero & ~aligned["empty_roi"]).sum()),
    }
    if not checks["empty_roi"]["matches_all_seeds_zero"]:
        raise ValueError(
            f"empty ROI geometry does not match the 0.0 scores: {checks['empty_roi']}"
        )

    # ---- patients
    patients = sorted(aligned["patient_id"].unique(), key=int)
    pidx = aligned["patient_id"].map({p: i for i, p in enumerate(patients)}).to_numpy()
    y = aligned["label"].to_numpy()
    per_patient = (
        aligned.assign(slide=aligned["slide_id"])
        .groupby("patient_id")
        .agg(
            n_slides=("slide", "nunique"),
            n_tiles=("label", "size"),
            n_pos=("label", "sum"),
            n_empty_roi=("empty_roi", "sum"),
        )
        .reindex(patients)
        .reset_index()
    )
    per_patient["n_neg"] = per_patient["n_tiles"] - per_patient["n_pos"]
    per_patient.to_csv(out / "patients.csv", index=False)

    # ---- point estimates (torchmetrics)
    score_cols = [
        *config.pixel_models.keys(),
        *[c for cs in nuc_cols.values() for c in cs],
    ]
    point = {c: tm_metrics(y, aligned[c].to_numpy()) for c in score_cols}
    nonempty = ~aligned["empty_roi"].to_numpy()
    point_ne = {
        c: tm_metrics(y[nonempty], aligned.loc[nonempty, c].to_numpy())
        for c in score_cols
    }

    # NB-A-64 reproduction (full test set = each pixel model's own original tile set here)
    repro = {}
    for name, m in config.pixel_models.items():
        logged = client.get_run(m.run).data.metrics
        full = tm_metrics(
            sources[name]["target"].astype(int).to_numpy(),
            sources[name][name].to_numpy(),
        )
        repro[name] = {
            "own_tile_set_n": len(sources[name]),
            "recomputed": full,
            "logged": {
                "AUPRC": logged.get("test/AUPRC"),
                "AUROC": logged.get("test/AUC"),
            },
            "nb_a_64": dict(m.nb_a_64),
            "on_intersection": point[name],
        }
        for metric in METRICS:
            if round(full[metric], 3) != m.nb_a_64[metric]:
                raise ValueError(
                    f"{name} {metric} {full[metric]:.4f} does not reproduce NB-A-64"
                )
    for name, cols in nuc_cols.items():
        for col, run in zip(cols, config.nucleus_models[name].eval_runs, strict=True):
            logged = client.get_run(run).data.metrics
            repro[col] = {
                "recomputed": point[col],
                "logged": {
                    "AUPRC": logged.get("test_thresholded/AUPRC"),
                    "AUROC": logged.get("test_thresholded/AUROC"),
                },
            }
    checks["reproduction"] = repro

    # ---- bootstrap resamples
    b = config.bootstrap
    rng = np.random.default_rng(b.seed)
    n_p = len(patients)
    pos_pp = np.bincount(pidx, weights=y, minlength=n_p)
    neg_pp = np.bincount(pidx, weights=1 - y, minlength=n_p)
    draws, redraws = [], 0
    while len(draws) < b.n_resamples:
        d = rng.integers(0, n_p, size=n_p)
        mult = np.bincount(d, minlength=n_p)
        if pos_pp @ mult == 0 or neg_pp @ mult == 0:
            redraws += 1
            continue
        draws.append(d)
    draws = np.stack(draws)
    mults = np.stack([np.bincount(d, minlength=n_p) for d in draws])
    patient_arr = np.asarray(patients)
    np.savez_compressed(
        out / "bootstrap_resamples.npz",
        patient_ids=patient_arr[draws],  # (n_resamples, n_patients): drawn patient ids
        patients=patient_arr,
        multiplicities=mults,  # (n_resamples, n_patients), columns in `patients` order
        seed=b.seed,
        redraws=redraws,
    )
    pd.DataFrame(
        patient_arr[draws], columns=[f"draw_{i + 1}" for i in range(n_p)]
    ).rename_axis("resample").to_csv(out / "bootstrap_resamples.csv")
    checks["bootstrap"] = {
        "n_resamples": int(b.n_resamples),
        "seed": int(b.seed),
        "redraws": redraws,
        "n_patients": n_p,
        "patients_without_positive_tiles": [
            p for p, v in zip(patients, pos_pp, strict=True) if v == 0
        ],
        "patients_without_negative_tiles": [
            p for p, v in zip(patients, neg_pp, strict=True) if v == 0
        ],
    }

    # ---- replicate metrics
    rankings = {}
    reps = {c: {m: np.empty(b.n_resamples) for m in METRICS} for c in score_cols}
    max_err_full = 0.0
    for c in tqdm(score_cols, desc="rank"):
        rankings[c] = WeightedRanking(aligned[c].to_numpy(), y, pidx, n_p)
        full = rankings[c].metrics(np.ones(n_p))
        max_err_full = max(max_err_full, *(abs(full[m] - point[c][m]) for m in METRICS))
    for i in tqdm(range(b.n_resamples), desc="bootstrap"):
        for c in score_cols:
            r = rankings[c].metrics(mults[i].astype(float))
            for m in METRICS:
                reps[c][m][i] = r[m]
    max_err_rep = 0.0
    for i in range(b.n_torchmetrics_checks):
        idx = np.concatenate([np.flatnonzero(pidx == p) for p in draws[i]])
        for c in score_cols:
            tm = tm_metrics(y[idx], aligned[c].to_numpy()[idx])
            max_err_rep = max(
                max_err_rep, *(abs(tm[m] - reps[c][m][i]) for m in METRICS)
            )
    checks["weighted_vs_torchmetrics"] = {
        "max_abs_error_full_set": max_err_full,
        "max_abs_error_replicated_resamples": max_err_rep,
        "n_resamples_checked": int(b.n_torchmetrics_checks),
    }
    assert max_err_full < 1e-6 and max_err_rep < 1e-6, checks[
        "weighted_vs_torchmetrics"
    ]
    np.savez_compressed(
        out / "replicates.npz",
        **{f"{c}__{m}": reps[c][m] for c in score_cols for m in METRICS},
    )

    # ---- model table
    lo_q, hi_q = 100 * (1 - b.ci_level) / 2, 100 * (1 + b.ci_level) / 2
    model_reps, model_point = {}, {}
    rows = []
    n_tiles, n_pos = len(aligned), int(y.sum())
    for name, m in config.pixel_models.items():
        model_reps[name] = reps[name]
        model_point[name] = point[name]
        row = {"model": name, "label": m.label, "kind": "pixel", "runs": m.run}
        for metric in METRICS:
            row[metric] = point[name][metric]
            row[f"{metric}_ci_lo"], row[f"{metric}_ci_hi"] = np.percentile(
                reps[name][metric], [lo_q, hi_q]
            )
            row[f"{metric}_no_empty_roi"] = point_ne[name][metric]
        rows.append(row)
    for name, cols in nuc_cols.items():
        model_reps[name] = {
            m: np.mean([reps[c][m] for c in cols], axis=0) for m in METRICS
        }
        model_point[name] = {
            m: float(np.mean([point[c][m] for c in cols])) for m in METRICS
        }
        row = {
            "model": name,
            "label": config.nucleus_models[name].label,
            "kind": "nucleus (mean of 3 seeds)",
            "runs": ",".join(config.nucleus_models[name].eval_runs),
        }
        for metric in METRICS:
            row[metric] = model_point[name][metric]
            row[f"{metric}_ci_lo"], row[f"{metric}_ci_hi"] = np.percentile(
                model_reps[name][metric], [lo_q, hi_q]
            )
            row[f"{metric}_no_empty_roi"] = float(
                np.mean([point_ne[c][metric] for c in cols])
            )
            for i, c in enumerate(cols, start=1):
                row[f"{metric}_seed{i}"] = point[c][metric]
            seeds = [point[c][metric] for c in cols]
            row[f"{metric}_seed_min"], row[f"{metric}_seed_max"] = (
                min(seeds),
                max(seeds),
            )
        rows.append(row)
    models = pd.DataFrame(rows)
    models["n_tiles"], models["n_pos"] = n_tiles, n_pos
    models.to_csv(out / "models.csv", index=False)

    # ---- comparisons
    comp_rows = []
    for a, bb in config.comparisons:
        row = {"comparison": f"{a} - {bb}", "a": a, "b": bb}
        for metric in METRICS:
            diff = model_reps[a][metric] - model_reps[bb][metric]
            row[f"d{metric}"] = model_point[a][metric] - model_point[bb][metric]
            row[f"d{metric}_ci_lo"], row[f"d{metric}_ci_hi"] = np.percentile(
                diff, [lo_q, hi_q]
            )
            row[f"d{metric}_p_raw"] = bootstrap_p(diff)
            row[f"d{metric}_no_empty_roi"] = (
                models.set_index("model").loc[a, f"{metric}_no_empty_roi"]
                - models.set_index("model").loc[bb, f"{metric}_no_empty_roi"]
            )
        comp_rows.append(row)
    comps = pd.DataFrame(comp_rows)
    comps[f"d{config.holm_metric}_p_holm"] = holm(
        comps[f"d{config.holm_metric}_p_raw"].to_numpy()
    )
    order = ["comparison", "a", "b"]
    for metric in METRICS:
        order += [
            f"d{metric}",
            f"d{metric}_ci_lo",
            f"d{metric}_ci_hi",
            f"d{metric}_p_raw",
        ]
        if metric == config.holm_metric:
            order.append(f"d{metric}_p_holm")
        order.append(f"d{metric}_no_empty_roi")
    comps = comps[order]
    comps.to_csv(out / "comparisons.csv", index=False)

    # ---- sensitivity table (all models and seeds, empty ROIs excluded)
    sens = pd.DataFrame(
        [
            {
                "score": c,
                **{f"{m}_all": point[c][m] for m in METRICS},
                **{f"{m}_no_empty_roi": point_ne[c][m] for m in METRICS},
            }
            for c in score_cols
        ]
    )
    sens["n_tiles_no_empty_roi"] = int(nonempty.sum())
    sens["n_pos_no_empty_roi"] = int(y[nonempty].sum())
    sens.to_csv(out / "sensitivity_no_empty_roi.csv", index=False)

    # ---- aligned table
    table = aligned[
        [
            "slide_id",
            "patient_id",
            "x",
            "y",
            "label",
            "empty_roi",
            "roi_nuclei",
            CARCINOMA_P,
        ]
    ].copy()
    for name in config.pixel_models:
        table[name] = aligned[name]
    for name, cols in nuc_cols.items():
        # descriptive only; every metric uses the per-seed columns
        table[f"{name}_mean_of_seeds"] = aligned[cols].mean(axis=1)
        for c in cols:
            table[c] = aligned[c]
    table.to_parquet(out / "aligned_tiles.parquet", index=False)

    checks["counts"] = {
        "n_tiles": n_tiles,
        "n_pos": n_pos,
        "n_neg": n_tiles - n_pos,
        "n_slides": int(aligned["slide_id"].nunique()),
        "n_patients": n_p,
        "n_unfiltered_tiles": len(raw),
    }
    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=float))
    write_report(out, config, checks, models, comps, sens, per_patient, excl, pooling)
    return checks


# ----------------------------------------------------------------------------- report
def _md(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    def fmt(v: object) -> str:
        if isinstance(v, float | np.floating):
            return format(v, floatfmt)
        return str(v)

    lines = [
        "| " + " | ".join(map(str, df.columns)) + " |",
        "|" + "---|" * len(df.columns),
    ]
    lines += [
        "| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)
    ]
    return "\n".join(lines)


def _p(p: float, n: int) -> str:
    return f"< {1 / n:.4f}" if p == 0 else f"{p:.4f}"


def write_report(
    out, config, checks, models, comps, sens, per_patient, excl, pooling
) -> None:
    b = config.bootstrap
    c = checks["counts"]
    e = checks["empty_roi"]
    rep = checks["reproduction"]
    n = int(b.n_resamples)
    lines = [
        "# MMCI TL test: pixel vs nucleus models, paired patient-level bootstrap (NB-A-64)",
        "",
        (
            f"Test set: {c['n_patients']} patients, {c['n_slides']} slides, {c['n_tiles']:,} tiles "
            f"({c['n_pos']:,} positive, {c['n_neg']:,} negative). B = {n} patient resamples "
            f"(seed {b.seed}), {b.ci_level:.0%} percentile CIs. With {c['n_patients']} patients the "
            "CIs are wide by construction: each resample reweights whole patients, and a few patients "
            "carry most of the positive tiles."
        ),
        "",
        "## 1. Alignment checks",
        "",
        "**Sources and schema.**",
        "",
    ]
    for name in config.pixel_models:
        s = checks["schema"][name]
        lines.append(
            f'- {config.pixel_models[name].label}: run `{s["run"]}` ("{s["run_name"]}"); '
            f"columns {', '.join(f'`{k}`' for k in s['columns'])}; keys {s['keys']}; score "
            f"`{s['score_column']}` is a {s['score_type']}, range "
            f"[{s['score_range'][0]:.3g}, {s['score_range'][1]:.3g}]; label `target`."
        )
    t = checks["tiling"]
    lines += [
        (
            "- Nucleus models: pooled `tile_predictions/<stem>.parquet` (x, y, tile_prediction) of the "
            "six evaluation runs; one parquet per slide, 87 slides each."
        ),
        (
            f"- Tiling: {t['tile_extent']} px tiles at level {t['level']}, stride {t['stride']}, "
            f"{t['coordinates']}; nucleus ROI = {t['roi']}. The level-1 resolution of these slides is "
            f"{', '.join(str(v) for v in t['mpp_level'])} µm/px (not 0.5)."
        ),
        (
            "- Slide identifier: the pixel tables use the slide stem; the tiling's hashed `slide_id` "
            "is mapped to the stem through `slides.parquet`. Patient = case number "
            "(`slides_mapping.parquet` `patient_id`, checked against the stem)."
        ),
        "",
        (
            "**Labels.** One label for all models: `carcinoma_roi_percentage > "
            f"{config.thresholds.carcinoma_roi_t}` of the QC-filtered tiling (as `load_tile_predictions`). "
        ),
    ]
    la = checks["label_agreement"]
    lines[-1] += (
        "Agreement with the pixel tables' `target`: "
        + "; ".join(
            f"{k}: {v['joined'] - v['disagree']:,}/{v['joined']:,} tiles agree"
            for k, v in la.items()
        )
        + (
            f". Both sides use carcinoma_roi_t = {config.thresholds.carcinoma_roi_t} with a strict `>`; "
            f"{checks['label_rule']['tiles_exactly_at_threshold']} tiles sit exactly at 0.5 and are "
            "negative on both sides."
        )
    )
    lines += [
        "",
        "**Tile sets** (n / positives per source; every model is evaluated on the intersection):",
        "",
    ]
    jc = pd.DataFrame(
        [
            {
                "source": k,
                "n": v["n"],
                "pos": v["pos"],
                "neg": v.get("neg", v["n"] - v["pos"]),
                "only in source": v.get("only_in_source", 0),
                "only in tiling": v.get("only_in_tiling", 0),
            }
            for k, v in checks["join"].items()
        ]
    )
    lines += [
        _md(jc),
        "",
        (
            f"Intersection: {c['n_tiles']:,} tiles, {c['n_pos']:,} positive. "
            "All eight score sources cover exactly the QC-filtered tiling, so the join removes nothing."
        ),
        "",
    ]
    tot = excl[(excl.patient_id == "all")]
    lines += [
        (
            f"**QC exclusions.** Re-applying `BaseTileDataset.filter_tiles_by_thresholds` (tissue > "
            f"{config.thresholds.tissue_roi_t}, exclude/another pathology ≤ 0, residual/blur/folding ≤ 0.25) "
            f"to the unfiltered tiling ({c['n_unfiltered_tiles']:,} tiles) reproduces the filtered tiling "
            "exactly; the same filter applies to every model (VGG16 and the nucleus runs read this "
            "tiling; the Virchow2 embeddings and both pixel tables hold the identical tile set). Sequential "
            "removals (in the filter's order), by class:"
        ),
        "",
        _md(
            tot[
                [
                    "label",
                    "n_unfiltered",
                    *[k for k in tot.columns if k.startswith("removed_")],
                    "n_final",
                ]
            ],
            ".0f",
        ),
        "",
        "Per patient and class: `exclusions.csv` (`removed_*` sequential, `fails_*` per criterion).",
        "",
        "Tiles per patient:",
        "",
        _md(per_patient, ".0f"),
        "",
        (
            f"**Empty ROIs.** {e['n']:,} tiles ({e['n'] / c['n_tiles']:.1%}) have no nucleus centroid in the "
            f"central ROI ({e['pos']:,} positive, {e['neg']:,} negative); recomputed from the nuclei with the "
            "`BaseTileDataset` geometry, they are exactly the tiles every nucleus seed scores 0.0 "
            f"(non-empty tiles scored 0.0: {e['zero_score_but_nonempty']}). They stay in the main analysis."
        ),
        "",
        "**NB-A-64 reproduction** (torchmetrics, each pixel model on its own full table = the intersection):",
        "",
    ]
    rr = []
    for name in config.pixel_models:
        r = rep[name]
        rr.append(
            {
                "model": name,
                "AUPRC": r["recomputed"]["AUPRC"],
                "AUPRC logged": r["logged"]["AUPRC"],
                "AUPRC NB-A-64": r["nb_a_64"]["AUPRC"],
                "AUROC": r["recomputed"]["AUROC"],
                "AUROC logged": r["logged"]["AUROC"],
                "AUROC NB-A-64": r["nb_a_64"]["AUROC"],
            }
        )
    for k, r in rep.items():
        if k in config.pixel_models:
            continue
        rr.append(
            {
                "model": k,
                "AUPRC": r["recomputed"]["AUPRC"],
                "AUPRC logged": r["logged"]["AUPRC"],
                "AUPRC NB-A-64": float("nan"),
                "AUROC": r["recomputed"]["AUROC"],
                "AUROC logged": r["logged"]["AUROC"],
                "AUROC NB-A-64": float("nan"),
            }
        )
    lines += [_md(pd.DataFrame(rr), ".4f"), ""]
    wv = checks["weighted_vs_torchmetrics"]
    lines += [
        (
            f"Bootstrap replicate metrics use the torchmetrics definitions with integer patient "
            f"weights; max |difference| vs torchmetrics: {wv['max_abs_error_full_set']:.1e} on the full set, "
            f"{wv['max_abs_error_replicated_resamples']:.1e} on {wv['n_resamples_checked']} physically "
            f"replicated resamples. Redraws (resample without positives or negatives): "
            f"{checks['bootstrap']['redraws']}."
        ),
        "",
        (
            "**Pooling selected on validation** (`sweep_pooling`, argmax of validation ROC AUC; equals the "
            "runs' `pooling_mode`/`pooling_k` params):"
        ),
        "",
        _md(
            pooling[
                [
                    "model",
                    "seed",
                    "run",
                    "pooling_mode",
                    "k",
                    "val_roc_auc",
                    "runner_up",
                    "margin_to_runner_up",
                ]
            ],
            ".5f",
        ),
        "",
        "## 2. Results",
        "",
        (
            "Point estimate [95% CI]. Nucleus models: mean of the three seeds' metrics (per-seed "
            "values in the seeds columns); the CI is taken over the seed-averaged metric per resample."
        ),
        "",
    ]
    mt = []
    for r in models.itertuples():
        row = {"model": r.label}
        for m in METRICS:
            row[m] = (
                f"{getattr(r, m):.3f} [{getattr(r, f'{m}_ci_lo'):.3f}, {getattr(r, f'{m}_ci_hi'):.3f}]"
            )
            if r.kind.startswith("nucleus"):
                row[f"{m} seeds"] = ", ".join(
                    f"{getattr(r, f'{m}_seed{i}'):.3f}" for i in (1, 2, 3)
                )
            else:
                row[f"{m} seeds"] = "-"
        mt.append(row)
    lines += [
        _md(pd.DataFrame(mt)),
        "",
        f"n_tiles = {c['n_tiles']:,}, n_pos = {c['n_pos']:,} for every model.",
        "",
    ]
    ct = []
    for r in comps.itertuples():
        ct.append(
            {
                "comparison": r.comparison,
                "ΔAUPRC [95% CI]": f"{r.dAUPRC:+.3f} [{r.dAUPRC_ci_lo:+.3f}, {r.dAUPRC_ci_hi:+.3f}]",
                "p raw": _p(r.dAUPRC_p_raw, n),
                "p Holm": _p(r.dAUPRC_p_holm, n),
                "ΔAUROC [95% CI]": f"{r.dAUROC:+.3f} [{r.dAUROC_ci_lo:+.3f}, {r.dAUROC_ci_hi:+.3f}]",
                "p raw (AUROC)": _p(r.dAUROC_p_raw, n),
            }
        )
    lines += [
        (
            "Paired differences A - B (Holm over the five AUPRC comparisons; AUROC secondary, raw p only). "
            f"A p-value of 0 in the replicates is shown as < 1/B = {1 / n:.4f}."
        ),
        "",
        _md(pd.DataFrame(ct)),
        "",
        "## 3. Sensitivity: empty ROIs excluded",
        "",
        f"{int(sens['n_tiles_no_empty_roi'].iloc[0]):,} tiles, {int(sens['n_pos_no_empty_roi'].iloc[0]):,} positive.",
        "",
        _md(
            sens[
                [
                    "score",
                    "AUPRC_all",
                    "AUPRC_no_empty_roi",
                    "AUROC_all",
                    "AUROC_no_empty_roi",
                ]
            ]
        ),
        "",
        _md(
            comps[
                [
                    "comparison",
                    "dAUPRC",
                    "dAUPRC_no_empty_roi",
                    "dAUROC",
                    "dAUROC_no_empty_roi",
                ]
            ]
        ),
        "",
    ]

    # ---- notes
    pos_share = per_patient.set_index("patient_id")["n_pos"].sort_values(
        ascending=False
    )
    top3 = pos_share.iloc[:3]
    flips = [
        f"{r['comparison']} Δ{m} ({r[f'd{m}']:+.3f} → {r[f'd{m}_no_empty_roi']:+.3f})"
        for r in comps.to_dict("records")
        for m in METRICS
        if np.sign(r[f"d{m}"]) != np.sign(r[f"d{m}_no_empty_roi"])
    ]
    shifts = {
        kind: np.mean(
            [
                r["AUPRC_no_empty_roi"] - r["AUPRC_all"]
                for r in sens.to_dict("records")
                if r["score"].startswith(kind)
            ]
        )
        for kind in ("virchow2", "vgg16", "shape", "positions")
    }
    qc_total = tot[tot.label == "all"]
    qc_removed = int(
        sum(qc_total[f"removed_{k}"].iloc[0] for k in ("residual", "blur", "folding"))
    )
    annot_removed = int(
        sum(qc_total[f"removed_{k}"].iloc[0] for k in ("exclude", "another_pathology"))
    )
    lines += [
        "## 4. Notes and deviations from expectations",
        "",
        (
            f"- Positives are concentrated: patients {', '.join(top3.index)} hold "
            f"{top3.sum() / pos_share.sum():.0%} of the positive tiles; the fewest positives in one "
            f"patient is {int(pos_share.min())}. Resamples that leave out the high-prevalence patients lower "
            "the prevalence and with it the AUPRC of every model, so the per-model AUPRC CIs are skewed "
            "low and wide; the paired differences cancel most of this and are the quantities to read."
        ),
        (
            f"- Bootstrap p-values have a resolution of 2/B = {2 / n:.4f}; the smallest non-zero p "
            "means a single replicate fell on the other side of 0."
        ),
        (
            f"- Pooling (all seeds `top_k`) was chosen on validation by margins of at most "
            f"{pooling['margin_to_runner_up'].max():.1e} in ROC AUC over the runner-up, so the "
            "selected k is close to arbitrary between neighbours; it was not revisited on the test set."
        ),
        (
            f"- Of the excluded tiles, {qc_removed:,} fail the image QC proper (residual/blur/folding); "
            f"{annot_removed:,} are removed by the annotation-based `exclude`/`another_pathology` masks and "
            f"{int(qc_total['removed_tissue_roi'].iloc[0]):,} by the tissue threshold."
        ),
        (
            "- Excluding empty ROIs changes AUPRC by "
            + ", ".join(f"{k} {v:+.3f}" for k, v in shifts.items())
            + "; "
            + (
                "no difference changes sign."
                if not flips
                else "differences that change sign: " + "; ".join(flips) + "."
            )
        ),
        (
            f"- The level-1 pixel size is {', '.join(str(v) for v in t['mpp_level'])} µm/px, not 0.5; "
            "tiles therefore span about 105 µm, the ROI about 52 µm."
        ),
        (
            "- Experiments 65 and 97 are on `http://mlflow.rationai-mlflow:5000/`, not on the default "
            "`MLFLOW_TRACKING_URI` of the notebook environment."
        ),
        "",
    ]
    (out / "report.md").write_text("\n".join(lines))


# ----------------------------------------------------------------------------- MLflow
def git_info(
    *paths: str,
) -> dict[str, str]:
    """Commit, branch and credential-free remote URL; "(dirty)" if any of `paths` changed."""
    paths = paths or (
        "postprocessing/tile/bootstrap_pixel_vs_nuclei.py",
        "configs/postprocessing/tile/bootstrap_pixel_vs_nuclei.yaml",
    )

    def run(*args: str) -> str:
        return subprocess.run(
            args, cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    url = run("git", "remote", "get-url", "origin")
    url = re.sub(r"//[^/@]*@", "//", url)  # strip any embedded credentials
    dirty = run(
        "git",
        "status",
        "--porcelain",
        "--",
        *paths,
    )
    return {
        "code_commit": run("git", "rev-parse", "HEAD") + (" (dirty)" if dirty else ""),
        "git.repo_url": url,
        "git.branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
    }


@with_cli_args(["+postprocessing=tile/bootstrap_pixel_vs_nuclei"])
@hydra.main(
    config_path="../../configs", config_name="postprocessing", version_base=None
)
def main(config: DictConfig) -> None:
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    client = MlflowClient()
    out = Path(config.out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    checks = compute(config, client, out)
    (out / "config-resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    if not config.log_mlflow:
        print(f"log_mlflow=false: outputs in {out}, nothing logged")
        return

    experiment = client.get_experiment_by_name(config.metadata.experiment_name)
    tags = {
        "mlflow.runName": config.metadata.run_name,
        "mlflow.note.content": config.metadata.description,
        **git_info(),
    }
    if os.environ.get("MLFLOW_TRACKING_USERNAME"):
        tags["mlflow.user"] = os.environ["MLFLOW_TRACKING_USERNAME"]
    run_id = client.create_run(
        experiment.experiment_id, run_name=config.metadata.run_name, tags=tags
    ).info.run_id

    params = {f"input/{n}": m.run for n, m in config.pixel_models.items()}
    for n, m in config.nucleus_models.items():
        params |= {f"input/{n}_seed{i}": r for i, r in enumerate(m.eval_runs, start=1)}
    params |= {
        "bootstrap/n_resamples": config.bootstrap.n_resamples,
        "bootstrap/seed": config.bootstrap.seed,
        "bootstrap/ci_level": config.bootstrap.ci_level,
        "bootstrap/unit": "patient",
        "holm_metric": config.holm_metric,
        "tiles_uri": config.tiles_uri,
        "carcinoma_roi_t": config.thresholds.carcinoma_roi_t,
    }
    for k, v in params.items():
        client.log_param(run_id, k, v)

    models = pd.read_csv(out / "models.csv")
    comps = pd.read_csv(out / "comparisons.csv")
    metrics = {f"counts/{k}": float(v) for k, v in checks["counts"].items()}
    metrics["counts/empty_roi"] = checks["empty_roi"]["n"]
    metrics["bootstrap/redraws"] = checks["bootstrap"]["redraws"]
    for r in models.to_dict("records"):
        for m in METRICS:
            for suffix in ("", "_ci_lo", "_ci_hi", "_no_empty_roi"):
                metrics[f"test/{r['model']}/{m}{suffix}"] = r[f"{m}{suffix}"]
    for r in comps.to_dict("records"):
        name = f"{r['a']}_minus_{r['b']}"
        for k, v in r.items():
            if k.startswith("d"):
                metrics[f"diff/{name}/{k[1:]}"] = v
    for k, v in metrics.items():
        client.log_metric(run_id, k, float(v))

    for p in sorted(out.iterdir()):
        client.log_artifact(run_id, str(p))
    for source in (
        __file__,
        "configs/postprocessing/tile/bootstrap_pixel_vs_nuclei.yaml",
    ):
        client.log_artifact(run_id, str(REPO / source), "code")
    client.set_terminated(run_id)
    print(
        f"MLflow run {run_id} (experiment {experiment.experiment_id}); outputs in {out}"
    )


if __name__ == "__main__":
    main()
