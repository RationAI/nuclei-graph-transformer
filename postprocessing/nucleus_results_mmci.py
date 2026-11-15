# ruff: noqa: RUF001  (Greek letters / minus signs in report text are intended)
r"""tab:res-nucleus, MMCI row: nucleus-level test results of the model on the MMCI TL test set.

The model: shape embedding (efd), rotated values (``attention_variant: vrope``), k-NN
block-sparse attention with k = 44, agreement supervision; three seeds (NB-A-65). Inference
outputs only: the per-nucleus test predictions stored in the three evaluation runs
(``<run>/artifacts/predictions``). No training, no inference.

Same method as the Radboud row (``postprocessing/nucleus_results.py`` at cb07dc8c, run
6aa01c04 on the public MLflow): the same label rule, exact torchmetrics metrics per seed, seed
mean and SD (ddof = 1), percentile CIs of the seed-averaged metric over B = 2,000 resamples, and
replicates from per-slide logit histograms on a uniform grid refined until every seed's
full-test-set metrics match the exact values within ``histogram.tolerance``. MMCI differences:

0. Runs. As the reference, plus: the expected training runs and the agreement supervision are
   asserted; the checkpoint path is parsed for any experiment. When the evaluation's saved config
   no longer holds the checkpoint (overwritten by a later postprocessing step on the same run),
   it is inferred, and marked as such, only if the training run holds a single checkpoint and the
   evaluation's re-validation metrics are closest to that checkpoint's epoch in the training
   history. The NB-A-64 (tile-level) evaluations of the same training runs are checked to rest on
   the same predictions: re-pooling the NB-A-65 predictions with each seed's pooling must
   reproduce their tile scores (``tile_check``).
1. Labels: the rule of ``postprocessing/crop/metrics`` (``annot_label`` left-joined on (slide,
   nucleus), missing -> 0, every nucleus of an ``is_carcinoma = False`` slide -> 0). The carcinoma
   annotation already excludes "Exclude" and "Another pathology" regions, so their nuclei are
   labelled 0; the ``another_pathology`` ignore labels are used by training supervision only and
   not by the test evaluation. Both are counted (ignore files; centroids in the annotation
   polygons), and no nucleus is ignored.
3. Bootstrap: the patient resamples of the tile-level bootstrap (run ``resamples_run``), reused
   unchanged; each nucleus is weighted by its patient's multiplicity. They hold real patient ids,
   so they are referenced by run id and sha256 and not re-logged.
4. Per patient (descriptive): P1-P10 from the private mapping of the tile-level breakdown.

Outputs (``out_dir``; all but the private inputs logged to one MLflow run in experiment 97):
``nucleus_results_mmci.csv`` (the columns of the Radboud ``nucleus_results.csv``),
``nucleus_per_patient_mmci.csv``, ``runs.csv``, ``checks.json``, ``report.md`` and
``replicates.npz``. Before logging every text output is checked for credential URLs and for
slide, patient and case ids; after logging, the run's tags and params are checked.
"""

import hashlib
import json
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import hydra
import mlflow
import numpy as np
import pandas as pd
import shapely
import torch
import yaml
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import with_cli_args
from ratiopath.openslide import OpenSlide
from ratiopath.parsers import ASAPParser
from shapely import MultiPolygon
from shapely.ops import unary_union
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_average_precision,
)

from nuclei_graph.data.datasets.tile.base import BaseTileDataset, get_slide_data
from postprocessing.bootstrap_sources import (
    bin_logits,
    hist_metrics,
    label_predictions,
    load_labels,
    load_run,
    replicate_metrics,
    slide_histograms,
    to_logits,
)
from postprocessing.tile.bootstrap_pixel_vs_nuclei import git_info


METRICS = ("AUPRC", "AUROC")
_PREFIXES = (
    "gh" + "p_",
    "github" + "_pat_",
    "gh" + "o_",
)  # split: not flagged when scanned
CREDENTIAL_URL = re.compile(r"https?://[^/\s\"']*@|" + "|".join(_PREFIXES))
CKPT = re.compile(
    r"mlflow-artifacts:/\d+/([0-9a-f]{32})/artifacts/checkpoints/epoch=(\d+)-step=\d+/"
)
REPO = Path(__file__).parents[1]
CODE = (
    "postprocessing/nucleus_results_mmci.py",
    "configs/postprocessing/nucleus_results_mmci.yaml",
)


# --------------------------------------------------------------------------
# 0. Runs
# --------------------------------------------------------------------------


def run_config(client: MlflowClient, run_id: str, tmp: Path) -> dict:
    (tmp / run_id).mkdir(parents=True, exist_ok=True)
    path = client.download_artifacts(
        run_id, "configs/config-resolved.yaml", str(tmp / run_id)
    )
    return yaml.safe_load(Path(path).read_text())


def model_config(cfg: dict) -> dict[str, str]:
    """The model configuration from a resolved run config (as the reference)."""
    net = cfg["model"]["net"]["config"]
    return {
        "embedding_mode": str(cfg["data"]["dataset"]["embedding_mode"]),
        "context": str(cfg["data"]["context"]),
        "k": str(cfg["data"]["k"]),
        "use_rope": str(net.get("use_rope", True)),
        "attention_variant": str(net.get("attention_variant", "standard")),
    }


def closest_epochs(
    client: MlflowClient, train_id: str, eval_metrics: dict, keys: list[str]
) -> list:
    """Training epochs ordered by max |training validation metric - evaluation re-validation|."""
    hist = {
        k: {m.step: m.value for m in client.get_metric_history(train_id, k)}
        for k in keys
    }
    epoch = {m.step: int(m.value) for m in client.get_metric_history(train_id, "epoch")}
    steps = sorted(set.intersection(*(set(h) for h in hist.values())))
    dist = [
        (max(abs(hist[k][s] - eval_metrics[k]) for k in keys), epoch.get(s, -1))
        for s in steps
    ]
    return sorted(dist)[:3]


def check_runs(config: DictConfig, client: MlflowClient, tmp: Path) -> pd.DataFrame:
    fix = pd.Timestamp(config.block_mask_fix_date, tz="UTC")
    rows = []
    for seed, eval_id in enumerate(config.eval_runs, start=1):
        ev = client.get_run(eval_id)
        train_id = ev.data.tags["mlflow.parentRunId"]
        tr = client.get_run(train_id)
        ev_cfg, tr_cfg = (
            run_config(client, eval_id, tmp),
            run_config(client, train_id, tmp),
        )

        monitors = [
            cb.get("monitor")
            for cb in tr_cfg["trainer"]["callbacks"]
            if isinstance(cb, dict)
            and cb.get("_target_", "").endswith("ModelCheckpoint")
        ]
        checkpoints = [f.path for f in client.list_artifacts(train_id, "checkpoints")]
        best = int(tr.data.metrics["best/epoch"])
        closest = closest_epochs(
            client, train_id, ev.data.metrics, list(config.revalidation_metrics)
        )
        ckpt = ev_cfg.get("checkpoint")
        if ckpt and CKPT.search(str(ckpt)):
            ckpt_run, ckpt_epoch = CKPT.search(str(ckpt)).groups()
            ckpt_epoch, source = int(ckpt_epoch), "saved evaluation config"
        else:
            ckpt_run, ckpt_epoch = (
                train_id,
                int(re.search(r"epoch=(\d+)", checkpoints[0]).group(1)),
            )
            source = "inferred: single checkpoint; re-validation closest to its epoch"
            assert len(checkpoints) == 1 and closest[0][1] == ckpt_epoch, (
                f"{eval_id}: checkpoint not recorded and not inferable"
            )
        has_model = "model" in ev_cfg
        eval_cfg = (
            model_config(ev_cfg)
            if has_model
            else {
                "embedding_mode": ev.data.params.get("embedding_mode"),
                "attention_variant": ev.data.params.get("attention_variant"),
            }
        )
        sup = tr_cfg["data"]["supervision"]
        rows.append(
            {
                "seed": seed,
                "training_run": train_id,
                "eval_run": eval_id,
                **model_config(tr_cfg),
                **{f"eval_{k}": v for k, v in eval_cfg.items()},
                "eval_config_source": "saved config"
                if has_model
                else "MLflow params (config overwritten)",
                "train_supervision": sup["train_strategy"]["mode"],
                "validation_supervision": sup["eval_strategy"]["mode"],
                "test_label_source": ev.data.params.get("prediction_supervision_mode"),
                "params_disagreeing_with_config": ",".join(
                    k
                    for k in config.expected_config
                    if k in tr.data.params
                    and tr.data.params[k] != model_config(tr_cfg)[k]
                ),
                "checkpoint_monitor": ",".join(map(str, monitors)),
                "training_split_size": tr_cfg["data"]["split_size"],
                "checkpoints_in_training_run": len(checkpoints),
                "checkpoint_run": ckpt_run,
                "checkpoint_epoch": ckpt_epoch,
                "checkpoint_source": source,
                "best_validation_epoch": best,
                "revalidation_closest_epoch": closest[0][1],
                "revalidation_max_abs_diff": closest[0][0],
                "revalidation_runner_up": f"epoch {closest[1][1]} ({closest[1][0]:.4f})",
                "best_validation_loss": tr.data.metrics.get("best/validation/loss"),
                "best_validation_AUPRC": tr.data.metrics.get("best/validation/AUPRC"),
                "last_logged_epoch": int(tr.data.metrics["epoch"]),
                "max_epochs": tr_cfg["trainer"]["max_epochs"],
                "training_status": tr.info.status,
                "trained_before_block_mask_fix": pd.Timestamp(
                    max(
                        m.timestamp
                        for m in client.get_metric_history(train_id, "epoch")
                    ),
                    unit="ms",
                    tz="UTC",
                )
                < fix,
                "evaluated_before_block_mask_fix": pd.Timestamp(
                    ev.info.start_time, unit="ms", tz="UTC"
                )
                < fix,
            }
        )
    runs = pd.DataFrame(rows)

    assert len(runs) == 3 and runs["training_run"].is_unique, (
        "need exactly 3 distinct seeds"
    )
    assert list(runs["training_run"]) == list(config.expected_training_runs), (
        "unexpected training runs"
    )
    assert (runs["params_disagreeing_with_config"] == "").all(), (
        "MLflow params contradict the config"
    )
    for key, value in config.expected_config.items():
        assert (runs[key] == str(value)).all(), f"training {key} != {value}"
        if f"eval_{key}" in runs:
            ok = runs[f"eval_{key}"].isna() | (
                runs[f"eval_{key}"].astype(str) == str(value)
            )
            assert ok.all(), f"evaluation {key} != {value}"
    assert (runs["train_supervision"] == config.expected_supervision).all()
    assert (runs["validation_supervision"] == config.expected_supervision).all()
    assert (runs["checkpoint_monitor"] == config.checkpoint_monitor).all()
    assert (runs["training_split_size"] > 0).all(), "no validation split"
    assert (runs["checkpoints_in_training_run"] == 1).all()
    assert (runs["checkpoint_run"] == runs["training_run"]).all(), (
        "evaluation loads another run's checkpoint"
    )
    assert (runs["checkpoint_epoch"] == runs["best_validation_epoch"]).all()
    assert (runs["revalidation_closest_epoch"] == runs["checkpoint_epoch"]).all()
    return runs


def _pool_slide(args: tuple) -> tuple[str, NDArray, int]:
    stem, xy, slide_path, nuclei_path, pred_paths, ks, tile_size, margin = args
    with OpenSlide(slide_path) as s:
        base, level = s.level_dimensions[0], s.level_dimensions[1]
    sx, sy = level[0] / base[0], level[1] / base[1]
    props = {
        "scale_x": sx,
        "scale_y": sy,
        "scaled_extent_x": tile_size / sx,
        "scaled_extent_y": tile_size / sy,
    }
    ns = SimpleNamespace(slide_props={stem: props}, tile_size=tile_size, margin=margin)
    _, cent, tree, ids = get_slide_data(nuclei_path)
    preds = [
        pd.read_parquet(p).set_index("id")["nuclei_prediction"].reindex(ids).to_numpy()
        for p in pred_paths
    ]
    missing = int(sum(np.isnan(p).sum() for p in preds))
    out = np.zeros((len(xy), len(ks)))
    for i, (x, y) in enumerate(xy):
        sp = BaseTileDataset.get_scaled_props(ns, {"stem": stem, "x": x, "y": y})
        idx = BaseTileDataset.get_tile_indices(ns, sp, cent, tree)
        if len(idx) == 0:
            continue
        roi = idx[BaseTileDataset.get_roi_mask(ns, sp, cent[idx])]
        for s, k in enumerate(
            ks
        ):  # top_k mean as pooling_utils.pool_predictions (float32)
            if len(roi):
                top = torch.topk(
                    torch.as_tensor(preds[s][roi], dtype=torch.float32),
                    min(k, len(roi)),
                )[0]
                out[i, s] = top.mean().item()
    return stem, out, missing


def tile_check(
    config: DictConfig, client: MlflowClient, meta: pd.DataFrame, pred_dirs: list[Path]
) -> dict:
    """NB-A-64 tile scores vs the NB-A-65 predictions pooled with each NB-A-64 seed's pooling."""
    tc = config.tile_check
    ks, modes = [], []
    for run in tc.eval_runs:
        p = client.get_run(run).data.params
        assert (
            client.get_run(run).data.tags["mlflow.parentRunId"]
            in config.expected_training_runs
        )
        modes.append(p["pooling_mode"])
        ks.append(int(p["pooling_k"]))
    assert set(modes) == {"top_k"}, "tile check implemented for top_k pooling"
    tiles = pd.read_parquet(
        download_artifacts(
            run_id=tc.aligned_tiles_run, artifact_path="aligned_tiles.parquet"
        )
    )
    m = meta.set_index("slide_id")
    jobs = [
        (
            stem,
            g[["x", "y"]].to_numpy(),
            m.loc[stem, "slide_path"],
            m.loc[stem, "slide_nuclei_path"],
            [d / f"{stem}.parquet" for d in pred_dirs],
            ks,
            tc.tile_size,
            tc.margin,
        )
        for stem, g in tiles.groupby("slide_id", sort=False)
    ]
    pooled, missing = np.zeros((len(tiles), len(ks))), 0
    with ProcessPoolExecutor(tc.workers) as ex:
        for stem, out, miss in ex.map(_pool_slide, jobs):
            pooled[(tiles["slide_id"] == stem).to_numpy()] = out
            missing += miss
    res = {"nuclei_without_prediction": missing, "n_tiles": len(tiles), "seeds": []}
    for s, (run, k) in enumerate(zip(tc.eval_runs, ks, strict=True)):
        d = np.abs(pooled[:, s] - tiles[f"shape_seed{s + 1}"].to_numpy())
        res["seeds"].append(
            {
                "nb_a_64_run": run,
                "pooling": f"top_k, k={k}",
                "max_abs_diff": float(d.max()),
                "tiles_above_tolerance": int((d > tc.tolerance).sum()),
                "mean_abs_diff": float(d.mean()),
            }
        )
    assert missing == 0 and all(
        r["tiles_above_tolerance"] == 0 for r in res["seeds"]
    ), res
    return res


# --------------------------------------------------------------------------
# 1. Labels and region counts
# --------------------------------------------------------------------------


def region_counts(config: DictConfig, meta: pd.DataFrame) -> dict:
    """Test nuclei in Exclude / Another pathology regions and their labels.

    Membership: centroid inside the annotation polygons; Another pathology also from the
    pipeline's ignore files.
    """
    rows = []
    for r in meta.itertuples():
        xml = Path(r.slide_path).with_suffix(config.annotation_suffix)
        if not xml.exists():
            continue
        parser = ASAPParser(xml)
        geoms = {}
        for group in ("Carcinoma", "Exclude", "Another pathology"):
            g = MultiPolygon(list(parser.get_polygons(part_of_group=group)))
            geoms[group] = unary_union(shapely.make_valid(g)) if not g.is_empty else g
        _, cent, _, ids = get_slide_data(str(r.slide_nuclei_path))
        inside = {
            k: shapely.contains_xy(g, cent[:, 0], cent[:, 1])
            if not g.is_empty
            else np.zeros(len(cent), bool)
            for k, g in geoms.items()
        }
        label = np.zeros(len(ids), np.int8)
        sup = Path(config.supervision_dir) / f"{r.slide_id}.parquet"
        if r.is_carcinoma and sup.exists():
            d = pd.read_parquet(sup, columns=["id", config.label_column]).set_index(
                "id"
            )[config.label_column]
            label = d.reindex(ids).fillna(0).to_numpy(np.int8)
        ign = Path(config.another_pathology_dir) / f"{r.slide_id}.parquet"
        ignore = (
            pd.read_parquet(ign)
            .set_index("id")["ignore_label"]
            .reindex(ids)
            .fillna(0)
            .to_numpy()
            == 1
            if ign.exists()
            else np.zeros(len(ids), bool)
        )
        rows.append(
            {
                "slide": r.slide_id,
                "annotated": True,
                "exclude": int(inside["Exclude"].sum()),
                "exclude_in_carcinoma": int(
                    (inside["Exclude"] & inside["Carcinoma"]).sum()
                ),
                "exclude_labelled_1": int((inside["Exclude"] & (label == 1)).sum()),
                "another": int(inside["Another pathology"].sum()),
                "another_in_carcinoma": int(
                    (inside["Another pathology"] & inside["Carcinoma"]).sum()
                ),
                "another_labelled_1": int(
                    (inside["Another pathology"] & (label == 1)).sum()
                ),
                "another_ignore_files": int(ignore.sum()),
                "another_ignore_files_labelled_1": int((ignore & (label == 1)).sum()),
                "has_exclude": not geoms["Exclude"].is_empty,
                "has_another": not geoms["Another pathology"].is_empty,
            }
        )
    df = pd.DataFrame(rows)
    out = {
        k: int(df[k].sum())
        for k in df.columns
        if k not in ("slide", "annotated", "has_exclude", "has_another")
    }
    out |= {
        "annotated_slides": len(df),
        "slides_with_exclude": int(df["has_exclude"].sum()),
        "slides_with_another_pathology": int(df["has_another"].sum()),
        "ignore_files_for_test_slides": int(
            sum(
                (Path(config.another_pathology_dir) / f"{s}.parquet").exists()
                for s in meta["slide_id"]
            )
        ),
    }
    return out


# --------------------------------------------------------------------------
# 2-4. Metrics, bootstrap, per patient
# --------------------------------------------------------------------------


def exact_metrics(y: NDArray, p: NDArray) -> dict[str, float]:
    preds, target = (
        torch.from_numpy(p.astype(np.float32)),
        torch.from_numpy(y.astype(np.int64)),
    )
    return {
        "AUPRC": float(binary_average_precision(preds, target, thresholds=None)),
        "AUROC": float(binary_auroc(preds, target, thresholds=None)),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute(config: DictConfig, client: MlflowClient) -> tuple[Path, dict, list[str]]:
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp"
    runs = check_runs(config, client, tmp)
    shutil.rmtree(tmp, ignore_errors=True)

    meta = pd.read_parquet(download_artifacts(config.metadata_uri)).sort_values(
        "slide_id", ignore_index=True
    )
    meta["patient_id"] = meta["patient_id"].astype(str)
    assert meta["slide_id"].is_unique
    slides = meta[["slide_id", "is_carcinoma"]]
    slide_ids = slides["slide_id"].tolist()
    carcinoma = slides["is_carcinoma"].to_numpy(bool)
    labels = load_labels(slides, Path(config.supervision_dir), config.label_column)
    assert set(np.unique(labels["label_sup"])) <= {0, 1}

    frames, pred_dirs = [], []
    for eval_id in config.eval_runs:
        df, info = load_run(
            eval_id, slide_ids, Path(config.cache_dir), config.predictions_artifact
        )
        assert info["test_slides_without_predictions"] == 0
        assert info["files_outside_test_metadata"] == 0
        frames.append(df)
        pred_dirs.append(
            Path(
                download_artifacts(
                    run_id=eval_id, artifact_path=config.predictions_artifact
                )
            )
        )
    ref = frames[0][["slide", "key"]]
    for f in frames[1:]:
        assert np.array_equal(f["slide"].to_numpy(), ref["slide"].to_numpy())
        assert np.array_equal(f["key"].to_numpy(), ref["key"].to_numpy()), (
            "seeds cover different nuclei"
        )
    y, unmatched = label_predictions(ref, labels, carcinoma)
    slide = ref["slide"].to_numpy()
    probs = np.stack([f["p"].to_numpy() for f in frames])
    n_slides = len(slides)

    tiles = tile_check(config, client, meta, pred_dirs)
    regions = region_counts(config, meta)

    positives_per_slide = np.bincount(slide, weights=(y == 1), minlength=n_slides)
    positive = positives_per_slide > 0
    counts = {
        "test_slides": n_slides,
        "carcinoma_slides": int(carcinoma.sum()),
        "positive_slides": int(positive.sum()),
        "negative_slides": int((~positive).sum()),
        "patients": int(meta["patient_id"].nunique()),
        "labeled_nuclei": len(y),
        "positive_nuclei": int((y == 1).sum()),
        "negative_nuclei": int((y == 0).sum()),
        "ignored_nuclei": 0,  # pipeline rule: Exclude / Another pathology nuclei are labelled 0
        "prevalence": float(np.mean(y == 1)),
        "carcinoma_slide_nuclei_without_supervision_row": unmatched,
    }
    assert np.bincount(slide, minlength=n_slides).min() > 0, (
        "a test slide has no nuclei"
    )

    exact = [exact_metrics(y, p) for p in probs]
    for i, e in enumerate(exact):
        runs.loc[i, [f"{m}_exact" for m in METRICS]] = [e[m] for m in METRICS]
    for i, run in enumerate(
        config.eval_runs
    ):  # the existing evaluation's logged values, where any
        logged = client.get_run(run).data.metrics
        for m in METRICS:
            runs.loc[i, f"{m}_logged_by_evaluation"] = logged.get(
                f"test_thresholded/nuclei_{m}"
            )

    # patient resamples of the tile-level bootstrap --------------------------------
    res_path = Path(
        download_artifacts(
            run_id=config.resamples_run, artifact_path=config.resamples_artifact
        )
    )
    r = np.load(res_path)
    assert int(r["seed"]) == int(config.resamples_seed)
    patients = [str(p) for p in r["patients"]]
    assert sorted(patients) == sorted(meta["patient_id"].unique()), (
        "resample patients != test patients"
    )
    mult = r["multiplicities"]
    assert (
        mult.shape == (len(r["patient_ids"]), len(patients))
        and (mult.sum(1) == len(patients)).all()
    )
    pidx = {p: i for i, p in enumerate(patients)}
    slide_patient = meta["patient_id"].map(pidx).to_numpy()
    weights = mult[:, slide_patient].astype(np.float64)  # (resamples, slides)
    pos_w = weights @ np.bincount(slide, weights=(y == 1), minlength=n_slides)
    neg_w = weights @ np.bincount(slide, weights=(y == 0), minlength=n_slides)
    assert (pos_w > 0).all() and (neg_w > 0).all(), (
        "a resample lacks positive or negative nuclei"
    )
    res_sha = sha256(res_path)

    # histograms, validated against the exact metrics ------------------------------
    logits = [to_logits(p) for p in probs]
    finite = np.concatenate([lg[np.isfinite(lg)] for lg in logits])
    lo, hi = float(finite.min()), float(finite.max())
    del finite
    n_bins = int(config.histogram.n_bins)
    while True:
        hists, errors = [], []
        for lg, e in zip(logits, exact, strict=True):
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
    for i, err in enumerate(errors):
        runs.loc[i, [f"{m}_hist_minus_exact" for m in METRICS]] = [
            err[m] for m in METRICS
        ]

    rep = {m: np.empty((len(probs), len(weights))) for m in METRICS}
    for i, (pos, neg) in enumerate(hists):
        for m, v in replicate_metrics(pos, neg, weights).items():
            rep[m][i] = v
    del hists
    assert all(np.isfinite(rep[m]).all() for m in METRICS)
    np.savez_compressed(
        out / "replicates.npz", eval_runs=np.array(list(config.eval_runs)), **rep
    )

    # results (columns of the Radboud nucleus_results.csv) --------------------------
    q = (1 - config.ci_level) / 2 * 100
    row = {
        "dataset": config.dataset_label,
        **{
            k: counts[k]
            for k in (
                "test_slides",
                "patients",
                "labeled_nuclei",
                "positive_nuclei",
                "ignored_nuclei",
                "prevalence",
            )
        },
    }
    for m in METRICS:
        per_seed = np.array([e[m] for e in exact])
        lo_ci, hi_ci = np.percentile(rep[m].mean(0), [q, 100 - q])
        row |= {
            m: float(per_seed.mean()),
            f"{m}_ci_lo": float(lo_ci),
            f"{m}_ci_hi": float(hi_ci),
            **{f"{m}_seed_{k + 1}": float(v) for k, v in enumerate(per_seed)},
            f"{m}_seed_sd": float(per_seed.std(ddof=1)),
        }
    results = pd.DataFrame([row])
    results.to_csv(out / "nucleus_results_mmci.csv", index=False)
    runs.to_csv(out / "runs.csv", index=False)

    # 4. per patient (P labels only) -------------------------------------------------
    mapping = pd.read_csv(config.patient_label_mapping, dtype=str)
    plabel = dict(zip(mapping["patient_id"], mapping["patient"], strict=True))
    assert set(plabel) == set(patients)
    nucleus_patient = slide_patient[slide]
    pp = []
    for p in sorted(patients, key=lambda p: int(plabel[p][1:])):
        sel = nucleus_patient == pidx[p]
        ys = y[sel]
        per = [exact_metrics(ys, pr[sel]) for pr in probs]
        pp.append(
            {
                "patient": plabel[p],
                "labeled_nuclei": int(sel.sum()),
                "positive_nuclei": int((ys == 1).sum()),
                "prevalence": float(ys.mean()),
                **{m: float(np.mean([e[m] for e in per])) for m in METRICS},
                **{
                    f"{m}_seed_{k + 1}": e[m]
                    for m in METRICS
                    for k, e in enumerate(per)
                },
            }
        )
    pp = pd.DataFrame(pp)
    pp.to_csv(out / "nucleus_per_patient_mmci.csv", index=False)

    checks = {
        "counts": counts,
        "labels": "annot_label left-joined on (slide_id, id), missing -> 0, is_carcinoma = False -> 0 "
        "(postprocessing/crop/metrics); carcinoma annotation = Carcinoma minus (Exclude ∪ Another "
        "pathology), so their nuclei are 0; no nucleus ignored",
        "regions": regions,
        "metrics": "torchmetrics binary_average_precision / binary_auroc (thresholds=None), exact, "
        "per seed over all labelled test nuclei pooled; seed mean and SD (ddof=1)",
        "resamples": {
            "run": config.resamples_run,
            "artifact": config.resamples_artifact,
            "sha256": res_sha,
            "seed": int(r["seed"]),
            "n_resamples": len(weights),
            "unit": "patient (all slides and nuclei of a drawn patient, with multiplicity)",
            "patients_match_test_patients": True,
            "relogged": False,
        },
        "histogram": {
            "n_bins": n_bins,
            "logit_range": [lo, hi],
            "tolerance": float(config.histogram.tolerance),
            "max_abs_error": max_err,
            "errors": dict(zip(config.eval_runs, errors, strict=True)),
        },
        "tile_check": tiles,
    }
    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=str))
    (out / "report.md").write_text(
        report(results.iloc[0], runs, counts, checks, pp, config)
    )
    private_ids = [*mapping["patient_id"], *mapping["case_id"], *slide_ids]
    return out, checks, private_ids


# --------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------


def report(
    res: pd.Series,
    runs: pd.DataFrame,
    counts: dict,
    checks: dict,
    pp: pd.DataFrame,
    config: DictConfig,
) -> str:
    ci = f"{config.ci_level:.0%}"
    rg, tc = checks["regions"], checks["tile_check"]
    lines = [
        f"# tab:res-nucleus: {config.dataset_label} test",
        "",
        (
            "Model: shape embedding (efd), rotated values (`vrope`), k-NN attention, k = 44, agreement "
            "supervision; three seeds (NB-A-65). Method of the Radboud row (`nucleus_results.py` at "
            "cb07dc8c, run 6aa01c04)."
        ),
        "",
        "## Step 0: runs",
        "",
        "| Seed | Training run | Evaluation run (NB-A-65) | Config (embedding, values, attention, k) | Supervision (train / val) | Checkpoint (best validation loss) | Checkpoint source |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in runs.itertuples():
        values = (
            "rotated values"
            if r.attention_variant == "vrope" and r.use_rope == "True"
            else r.attention_variant
        )
        lines.append(
            f"| {r.seed} | `{r.training_run}` | `{r.eval_run}` | {r.embedding_mode}, {values}, {r.context}, {r.k} | "
            f"{r.train_supervision} / {r.validation_supervision} | epoch {r.checkpoint_epoch} | {r.checkpoint_source} |"
        )
    lines += [
        "",
        f"Checkpoints are selected on `{config.checkpoint_monitor}` of the validation split of the training "
        "metadata only (ModelCheckpoint, save_top_k = 1); each training run holds one checkpoint, at its best "
        "validation epoch. Every evaluation's re-validation metrics are closest to that epoch in the training "
        "history (max |Δ| "
        + ", ".join(f"{v:.4f}" for v in runs["revalidation_max_abs_diff"])
        + "; runner-up "
        + "; ".join(runs["revalidation_runner_up"])
        + ").",
        f"Trained before the block-mask fix ({config.block_mask_fix_date}): "
        + ", ".join(str(v) for v in runs["trained_before_block_mask_fix"])
        + "; evaluated before it: "
        + ", ".join(str(v) for v in runs["evaluated_before_block_mask_fix"])
        + ".",
        "",
        "Only the NB-A-65 runs hold per-nucleus test predictions; the NB-A-64 runs (`"
        + "`, `".join(r["nb_a_64_run"] for r in tc["seeds"])
        + "`) hold pooled tile scores only. Re-pooling the "
        "NB-A-65 predictions with each NB-A-64 seed's validation-selected pooling reproduces their "
        f"{tc['n_tiles']:,} tile scores: max |Δ| "
        + ", ".join(f"{s['max_abs_diff']:.1e}" for s in tc["seeds"])
        + " (pooling "
        + ", ".join(s["pooling"] for s in tc["seeds"])
        + "), so both sets rest on the same "
        "predictions and checkpoints.",
        "",
        "## Step 1: labels",
        "",
        (
            "Dense annotation labels as the existing MMCI evaluation (`postprocessing/crop/metrics`): `annot_label` "
            "left-joined on (slide, nucleus), missing → 0, all nuclei of benign slides → 0. The carcinoma annotation "
            "is Carcinoma minus (Exclude ∪ Another pathology), so nuclei in Exclude and Another pathology regions are "
            "**labelled negative, not ignored**; the Another pathology ignore labels are used by the training "
            "supervision only."
        ),
        "",
        "| Test slides | Patients | Labelled nuclei | Positive nuclei | Ignored nuclei | Prevalence |",
        "| --- | --- | --- | --- | --- | --- |",
        (
            f"| {counts['test_slides']} | {counts['patients']} | {counts['labeled_nuclei']:,} | "
            f"{counts['positive_nuclei']:,} | {counts['ignored_nuclei']} | {counts['prevalence']:.4f} |"
        ),
        "",
        f"Carcinoma slides {counts['carcinoma_slides']}; slides with ≥ 1 positive nucleus {counts['positive_slides']}.",
        f"Region counts ({rg['annotated_slides']} annotated test slides; centroid inside the annotation polygons):",
        (
            f"- Exclude: {rg['exclude']:,} nuclei on {rg['slides_with_exclude']} slides, of which "
            f"{rg['exclude_in_carcinoma']:,} inside a Carcinoma polygon (labelled 0 because of the exclusion); "
            f"labelled 1: {rg['exclude_labelled_1']:,}."
        ),
        (
            f"- Another pathology: {rg['another']:,} nuclei on {rg['slides_with_another_pathology']} slides, of which "
            f"{rg['another_in_carcinoma']:,} inside a Carcinoma polygon; labelled 1: {rg['another_labelled_1']:,}. "
            f"The pipeline's ignore files ({rg['ignore_files_for_test_slides']} test slides) mark "
            f"{rg['another_ignore_files']:,} nuclei (labelled 1: {rg['another_ignore_files_labelled_1']:,})."
        ),
        "",
        "## Steps 2-3: results",
        "",
        f"| Metric | Seed mean [{ci} CI] | Seeds | SD over seeds |",
        "| --- | --- | --- | --- |",
    ]
    for m in METRICS:
        seeds = ", ".join(f"{res[f'{m}_seed_{k}']:.4f}" for k in (1, 2, 3))
        lines.append(
            f"| {m} | {res[m]:.4f} [{res[f'{m}_ci_lo']:.4f}, {res[f'{m}_ci_hi']:.4f}] | {seeds} | {res[f'{m}_seed_sd']:.4f} |"
        )
    h, rs = checks["histogram"], checks["resamples"]
    logged = runs[[f"{m}_logged_by_evaluation" for m in METRICS]].dropna()
    lines += [
        "",
        (
            f"CIs: percentile, B = {rs['n_resamples']:,} patient resamples of the tile-level bootstrap (run "
            f"`{rs['run']}`, `{rs['artifact']}`, seed {rs['seed']}, sha256 {rs['sha256'][:16]}…, not re-logged: it "
            "holds real patient ids), seed-averaged metric per resample. With 10 patients the CIs are wide by "
            "construction."
        ),
        (
            f"Histogram check: {h['n_bins']:,} bins over the logit range; max |histogram − exact torchmetrics| = "
            f"{h['max_abs_error']:.1e} (tolerance {h['tolerance']:.0e})."
        ),
    ]
    if len(logged):
        r0 = runs.loc[logged.index[0]]
        lines.append(
            f"The existing evaluation logged seed {r0['seed']}'s AUPRC/AUROC as "
            f"{r0['AUPRC_logged_by_evaluation']:.4f} / {r0['AUROC_logged_by_evaluation']:.4f}; recomputed: "
            f"{r0['AUPRC_exact']:.4f} / {r0['AUROC_exact']:.4f}."
        )
    lines += [
        "",
        "## Step 4: per patient (descriptive, mean of seeds)",
        "",
        "| Patient | Labelled nuclei | Positive nuclei | Prevalence | AUPRC | AUROC |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in pp.itertuples():
        lines.append(
            f"| {r.patient} | {r.labeled_nuclei:,} | {r.positive_nuclei:,} | {r.prevalence:.4f} | "
            f"{r.AUPRC:.4f} | {r.AUROC:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def preflight(out: Path, private_ids: list[str]) -> None:
    """No credentials in any text output; no slide, patient or case ids."""
    ids = re.compile(
        r"(?<![\w.,])("
        + "|".join(map(re.escape, private_ids))
        + r")(?![\w.,])|TP-\d{4}_"
    )
    for path in out.iterdir():
        if path.suffix not in {".csv", ".json", ".md", ".yaml"}:
            continue
        text = path.read_text()
        assert not CREDENTIAL_URL.search(text), f"credential URL in {path.name}"
        hit = ids.search(text)
        assert hit is None, f"private id in {path.name}"


@with_cli_args(["+postprocessing=nucleus_results_mmci"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
def main(config: DictConfig) -> None:
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    client = MlflowClient()
    out, checks, private_ids = compute(config, client)
    (out / "config-resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    preflight(out, private_ids)
    if not config.log_mlflow:
        print(f"log_mlflow=false: outputs in {out}")
        return

    runs = pd.read_csv(out / "runs.csv")
    params = {
        "dataset": config.dataset_label,
        "resamples_run": config.resamples_run,
        "resamples_sha256": checks["resamples"]["sha256"],
        "resamples_seed": config.resamples_seed,
        "ci_level": config.ci_level,
        "histogram_tolerance": config.histogram.tolerance,
        "n_eval_runs": len(runs),
        "reference_run": "6aa01c0448ec42a49ae5137ceaf4f12e (public MLflow, exp 37)",
        "reference_code_commit": "cb07dc8c5f15e57658081e8e53d236a2fd7ee869",
    }
    for r in runs.itertuples():
        params |= {
            f"eval_run_{r.seed}": r.eval_run,
            f"training_run_{r.seed}": r.training_run,
        }
    for i, run in enumerate(config.tile_check.eval_runs, start=1):
        params[f"tile_eval_run_{i}"] = run
    params["aligned_tiles_run"] = config.tile_check.aligned_tiles_run

    tags = {
        "mlflow.runName": config.metadata.run_name,
        "mlflow.note.content": config.metadata.description,
        **git_info(*CODE),
    }
    for text in [*map(str, tags.values()), *map(str, params.values())]:
        assert not CREDENTIAL_URL.search(text), "credential URL in a tag or param"
    experiment = client.get_experiment_by_name(config.metadata.experiment_name)
    run_id = client.create_run(
        experiment.experiment_id, run_name=config.metadata.run_name, tags=tags
    ).info.run_id
    user = client.get_run(config.eval_runs[0]).data.tags.get("mlflow.user")
    if user:
        client.set_tag(run_id, "mlflow.user", user)
    for k, v in params.items():
        client.log_param(run_id, k, v)
    res = pd.read_csv(out / "nucleus_results_mmci.csv").iloc[0]
    for k in res.index:
        if k not in {"dataset"} and pd.notna(res[k]):
            client.log_metric(run_id, f"test/{k}", float(res[k]))
    client.log_metric(
        run_id, "histogram/max_abs_error", checks["histogram"]["max_abs_error"]
    )
    for name in (
        "nucleus_results_mmci.csv",
        "nucleus_per_patient_mmci.csv",
        "runs.csv",
        "checks.json",
        "report.md",
        "replicates.npz",
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
        f"MLflow run {run_id} (experiment {experiment.experiment_id}); outputs in {out}"
    )


if __name__ == "__main__":
    main()
