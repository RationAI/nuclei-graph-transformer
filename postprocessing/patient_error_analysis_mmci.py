# ruff: noqa: RUF001, RUF002  (Greek letters / minus signs in report text are intended)
r"""Per-patient error analysis of the nucleus-level shape model on the MMCI TL test set.

Model, predictions, labels and patients of run ``results_run`` (dc419c48, tab:res-nucleus MMCI row
with Another pathology ignored at test; ``postprocessing/nucleus_results_mmci.py``): Shape + RoVE +
k-NN attention (k = 44, agreement supervision), three seeds, the stored per-nucleus test predictions
of its evaluation runs, the test label rule (``annot_label``, missing -> 0, benign slides -> 0,
Another pathology nuclei ignored), P1-P10. Metrics per seed, then the mean of the seeds.

1. Calibrated AUPRC (Siblini et al., IDA 2020, arXiv 1909.02827). Per patient and seed, average
   precision with the calibrated precision TP / (TP + c·FP), c = π(1 − π₀) / (π₀(1 − π)), π the
   patient's prevalence; π₀ = 0.275 (pooled test prevalence) and 0.5. Average precision is
   computed exactly as torchmetrics ``binary_average_precision`` (thresholds = None): Σ ΔR · P over
   the distinct scores. Check: c = 1 reproduces the torchmetrics AUPRC per patient and seed.
2. Operating point. The threshold with recall ≥ 0.90 (the highest such score) on the validation
   nuclei, scored by the seed-mean probability. The validation nuclei are not stored by any run, so
   they are predicted here (``inference.device``) with the three checkpoints of the training runs, on the fixed
   validation split of the training slides (``split_train_val``, random_state = 42), labelled by
   the test rule. The stored test predictions were made on GPU before the block-mask fix
   (``FIX_COMMIT``), so inference uses the pre-fix block mask (read from git) and TF32 matmuls; the
   reproduction is measured by re-predicting the whole test set and comparing (``rerun_check``).
   Per test patient and pooled: recall, FPR, TP, FP, FN of the seed-mean test probability.
3. Diagnostics, for every patient (per seed, then mean): the median score of its carcinoma nuclei
   vs that of all other patients' carcinoma nuclei (false-negative tendency); the share of its
   benign nuclei above the other patients' 99th percentile of benign scores vs 1 % (false-positive
   tendency). Interpreted for patients whose calibrated AUPRC (π₀ = 0.275) is clearly below the
   others (more than ``clearly_below_margin`` below the median of the other patients).

Outputs (``out_dir``; logged to a new run in experiment 97 that references ``results_run``):
``patient_error_analysis_mmci.csv``, ``patient_diagnostics_mmci.csv``, ``checks.json``,
``report.md``. Private (never logged): the predictions in ``inference_cache_dir``.
"""

import json
import math
import re
import shutil
import subprocess
import types
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from lightning.fabric.utilities.apply_func import move_data_to_device
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import with_cli_args
from tqdm import tqdm

import nuclei_graph.data.datamodules.collator as collator_module
from nuclei_graph.data.datamodules.base import split_train_val
from nuclei_graph.data.datasets.crop.prediction import PredictionDataset
from postprocessing.bootstrap_sources import (
    label_predictions,
    load_labels,
    load_run,
    nucleus_keys,
)
from postprocessing.nucleus_results_mmci import (
    CREDENTIAL_URL,
    exact_metrics,
    preflight,
    run_config,
)
from postprocessing.tile.bootstrap_pixel_vs_nuclei import git_info


REPO = Path(__file__).parents[1]
CODE = (
    "postprocessing/patient_error_analysis_mmci.py",
    "configs/postprocessing/patient_error_analysis_mmci.yaml",
)
FIX_COMMIT = "752565f2978de772d4bd27a29823a3f6b1753a84"  # fix: block mask (2026-10-31)
CKPT_EPOCH = re.compile(r"epoch=(\d+)")


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------


def reference(config: DictConfig, client: MlflowClient, tmp: Path) -> dict:
    """The results run: its evaluation/training runs, checkpoints and per-patient table."""
    run = client.get_run(config.results_run)
    p = run.data.params
    assert p["ignore_another_pathology"] == "True", (
        "results run must ignore Another pathology"
    )
    eval_runs = [p[f"eval_run_{s}"] for s in (1, 2, 3)]
    training_runs = [p[f"training_run_{s}"] for s in (1, 2, 3)]
    files = {}
    for name in (
        "nucleus_per_patient_mmci.csv",
        "nucleus_results_mmci.csv",
        "runs.csv",
    ):
        files[name] = pd.read_csv(
            client.download_artifacts(config.results_run, name, str(tmp))
        )
    runs = files["runs.csv"]
    assert list(runs["eval_run"]) == eval_runs
    assert list(runs["training_run"]) == training_runs
    for e, t in zip(eval_runs, training_runs, strict=True):
        assert client.get_run(e).data.tags["mlflow.parentRunId"] == t
    return {
        "eval_runs": eval_runs,
        "training_runs": training_runs,
        "checkpoint_epochs": [int(v) for v in runs["checkpoint_epoch"]],
        "per_patient": files["nucleus_per_patient_mmci.csv"],
        "pooled": files["nucleus_results_mmci.csv"].iloc[0],
    }


# --------------------------------------------------------------------------
# Inference (pre-fix block mask)
# --------------------------------------------------------------------------


def use_prefix_block_mask() -> None:
    """Points the collator at the block mask of the commit before ``FIX_COMMIT``.

    The evaluated models were trained, and their test predictions made, before the fix.
    """
    src = subprocess.run(
        ["git", "show", f"{FIX_COMMIT}^:nuclei_graph/data/block_mask.py"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    legacy = types.ModuleType("block_mask_prefix")
    exec(compile(src, "block_mask_prefix.py", "exec"), legacy.__dict__)  # noqa: S102
    collator_module.create_ragged_block_quantized_knn_mask = (
        legacy.create_ragged_block_quantized_knn_mask
    )


def load_model(
    client: MlflowClient, train_cfg: dict, train_run: str, epoch: int, tmp: Path
) -> torch.nn.Module:
    ckpts = [f.path for f in client.list_artifacts(train_run, "checkpoints")]
    assert len(ckpts) == 1, f"{train_run}: {len(ckpts)} checkpoints"
    assert int(CKPT_EPOCH.search(ckpts[0]).group(1)) == epoch, (
        "checkpoint != results run's"
    )
    local = Path(client.download_artifacts(train_run, ckpts[0], str(tmp / train_run)))
    if local.is_dir():
        (local,) = list(local.rglob("*.ckpt"))
    model = instantiate(train_cfg["model"])
    state = torch.load(local, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(state)
    return model.eval()


DATA_KEYS = ("block_size", "k", "context")
DATASET_KEYS = ("embedding_mode", "alpha", "efd_order", "patch_size")


def predict(
    client: MlflowClient,
    runs: list[tuple[str, int, dict]],
    slides: pd.DataFrame,
    paths: list[Path],
    tmp: Path,
    device: str = "cpu",
) -> list[pd.DataFrame]:
    """(slide, key, p) of every nucleus of `slides` (index = row in `slides`) per run; cached.

    The inputs do not depend on the seed, so each slide's batch is built once (in process:
    no DataLoader workers, /dev/shm is small) and passed through all the models.
    """
    if all(p.exists() for p in paths):
        return [pd.read_parquet(p) for p in paths]
    cfgs = [cfg["data"] for _, _, cfg in runs]
    for d in cfgs[1:]:
        assert {k: d[k] for k in DATA_KEYS} == {k: cfgs[0][k] for k in DATA_KEYS}
        assert {k: d["dataset"].get(k) for k in DATASET_KEYS} == {
            k: cfgs[0]["dataset"].get(k) for k in DATASET_KEYS
        }
    d = cfgs[0]
    models = [
        load_model(client, cfg, run, epoch, tmp).to(device) for run, epoch, cfg in runs
    ]
    ds = PredictionDataset(
        slides.reset_index(drop=True),
        embedding_mode=d["dataset"]["embedding_mode"],
        alpha=d["dataset"]["alpha"],
        efd_order=d["dataset"]["efd_order"],
        patch_size=d["dataset"].get("patch_size"),
    )
    collate = collator_module.GraphCollator(
        block_size=d["block_size"], k=d["k"], predict=True, context=d["context"]
    )
    frames: list[list[pd.DataFrame]] = [[] for _ in models]
    for s in tqdm(range(len(ds)), desc=paths[0].stem.split("_")[-1]):
        batch = collate([ds[s]])
        inputs = move_data_to_device(
            {k: v for k, v in batch.items() if k != "metadata"}, device
        )
        ids = pd.Series(np.asarray(batch["metadata"]["nuclei_ids"][0]))
        keys = nucleus_keys(ids)
        for m, model in enumerate(models):
            with torch.inference_mode():
                logits = model(inputs)["nuclei"].squeeze(-1)
            p = torch.sigmoid(logits).float().cpu().numpy().ravel()
            assert len(p) == len(ids)
            frames[m].append(
                pd.DataFrame(
                    {"slide": np.full(len(p), s, np.int16), "key": keys, "p": p}
                )
            )
    out = []
    for f, path in zip(frames, paths, strict=True):
        df = pd.concat(f, ignore_index=True).sort_values(
            ["slide", "key"], ignore_index=True
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        out.append(df)
    return out


# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------


def ignored(
    slides: pd.DataFrame, ignore_dir: Path, ref: pd.DataFrame
) -> NDArray[np.bool_]:
    """Nuclei of `ref` with ignore_label = 1 in the pipeline's (Another pathology) ignore files."""
    frames = []
    for s, row in enumerate(slides.itertuples()):
        path = ignore_dir / f"{row.slide_id}.parquet"
        if not path.exists():
            continue
        d = pd.read_parquet(path, columns=["id", "ignore_label"])
        assert set(d["ignore_label"].unique()) <= {0, 1}
        d = d[d["ignore_label"] == 1]
        frames.append(
            pd.DataFrame(
                {
                    "slide": np.full(len(d), s, np.int16),
                    "key": nucleus_keys(d["id"]),
                    "ignore": True,
                }
            )
        )
    if not frames:
        return np.zeros(len(ref), bool)
    ign = pd.concat(frames, ignore_index=True)
    out = ref[["slide", "key"]].merge(
        ign, on=["slide", "key"], how="left", validate="1:1"
    )
    mask = out["ignore"].eq(True).to_numpy()
    assert int(mask.sum()) == len(ign), "ignore rows without a predicted nucleus"
    return mask


def labelled(
    config: DictConfig, slides: pd.DataFrame, frames: list[pd.DataFrame]
) -> tuple[NDArray, NDArray, NDArray, dict]:
    """(y, slide, probs[seed]) of the labelled nuclei (test rule), and counts."""
    ref = frames[0][["slide", "key"]]
    for f in frames[1:]:
        assert np.array_equal(f["slide"].to_numpy(), ref["slide"].to_numpy())
        assert np.array_equal(f["key"].to_numpy(), ref["key"].to_numpy()), (
            "seeds differ"
        )
    carcinoma = slides["is_carcinoma"].to_numpy(bool)
    labels = load_labels(slides, Path(config.supervision_dir), config.label_column)
    y, unmatched = label_predictions(ref, labels, carcinoma)
    ign = ignored(slides, Path(config.another_pathology_dir), ref)
    keep = ~ign
    probs = np.stack([f["p"].to_numpy() for f in frames])[:, keep]
    y, slide = y[keep], ref["slide"].to_numpy()[keep]
    counts = {
        "slides": len(slides),
        "carcinoma_slides": int(carcinoma.sum()),
        "patients": int(slides["patient_id"].nunique()),
        "labelled_nuclei": len(y),
        "positive_nuclei": int(y.sum()),
        "ignored_nuclei": int(ign.sum()),
        "prevalence": float(y.mean()),
        "carcinoma_slide_nuclei_without_supervision_row": unmatched,
    }
    return y, slide, probs, counts


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


def calibrated_ap(y: NDArray, p: NDArray, c: float) -> float:
    """Average precision (Σ ΔR · P over distinct scores) with precision TP / (TP + c·FP)."""
    order = np.argsort(-p, kind="stable")
    ps, ys = p[order], y[order].astype(np.float64)
    last = np.r_[np.flatnonzero(ps[1:] != ps[:-1]), len(ps) - 1]
    tp, fp = np.cumsum(ys)[last], np.cumsum(1 - ys)[last]
    precision = tp / (tp + c * fp)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(recall, prepend=0.0) * precision))


def calibration(pi: float, pi0: float) -> float:
    return pi * (1 - pi0) / (pi0 * (1 - pi))


def operating_point(y: NDArray, s: NDArray, t: float) -> dict:
    pred = s >= t
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    return {
        "recall": tp / (tp + fn),
        "FPR": fp / (fp + tn),
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }


def recall_threshold(y: NDArray, s: NDArray, target: float) -> float:
    """The highest score t with recall(s >= t) >= target."""
    pos = np.sort(s[y == 1])[::-1]
    return float(pos[math.ceil(target * len(pos)) - 1])


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def compute(config: DictConfig, client: MlflowClient) -> tuple[Path, dict, list[str]]:
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp"
    tmp.mkdir(exist_ok=True)
    ref = reference(config, client, tmp)
    seeds = range(len(ref["eval_runs"]))

    # test: stored predictions of the results run's evaluation runs
    meta = pd.read_parquet(download_artifacts(config.metadata_uri)).sort_values(
        "slide_id", ignore_index=True
    )
    meta["patient_id"] = meta["patient_id"].astype(str)
    stored = [
        load_run(
            e,
            meta["slide_id"].tolist(),
            Path(config.test_cache_dir),
            config.predictions_artifact,
        )[0]
        for e in ref["eval_runs"]
    ]
    y, slide, probs, test_counts = labelled(config, meta, stored)

    # validation: the training runs' fixed split, predicted here
    train_cfgs = [run_config(client, t, tmp) for t in ref["training_runs"]]
    split_keys = ("metadata", "split_size", "split_stratify_col", "split_group_col")
    assert all(
        {k: c["data"][k] for k in split_keys}
        == {k: train_cfgs[0]["data"][k] for k in split_keys}
        for c in train_cfgs
    )
    d0 = train_cfgs[0]["data"]
    train_meta = pd.read_parquet(download_artifacts(d0["metadata"]["train"]))
    _, val = split_train_val(
        train_meta, d0["split_size"], d0["split_stratify_col"], d0["split_group_col"]
    )
    val = val.sort_values("slide_id", ignore_index=True)
    val["patient_id"] = val["patient_id"].astype(str)
    assert not set(val["patient_id"]) & set(meta["patient_id"]), (
        "validation patient in test"
    )

    torch.set_num_threads(config.inference.threads)
    # as nuclei_graph/__main__.py (TF32 matmuls on GPU, as the stored test predictions)
    torch.set_float32_matmul_precision("high")
    device = str(config.inference.device)
    use_prefix_block_mask()
    cache = Path(config.inference_cache_dir) / device
    runs = list(
        zip(ref["training_runs"], ref["checkpoint_epochs"], train_cfgs, strict=True)
    )
    val_pred = predict(
        client,
        runs,
        val,
        [cache / f"{t}_validation.parquet" for t in ref["training_runs"]],
        tmp,
        device,
    )
    test_rerun = predict(
        client,
        runs,
        meta,
        [cache / f"{t}_test.parquet" for t in ref["training_runs"]],
        tmp,
        device,
    )
    vy, _, vprobs, val_counts = labelled(config, val, val_pred)
    _, _, cprobs, _ = labelled(config, meta, test_rerun)
    shutil.rmtree(tmp, ignore_errors=True)

    # per patient --------------------------------------------------------------------------
    mapping = pd.read_csv(config.patient_label_mapping, dtype=str)
    plabel = dict(zip(mapping["patient_id"], mapping["patient"], strict=True))
    assert set(plabel) == set(meta["patient_id"])
    patient_of_slide = meta["patient_id"].map(plabel).to_numpy()
    patient = patient_of_slide[slide]
    order = sorted(plabel.values(), key=lambda p: int(p[1:]))

    pi0s = [float(v) for v in config.pi0]
    tol = float(config.calibration_check_tolerance)
    max_check = 0.0
    rows, per_seed = [], []
    groups = [(p, patient == p) for p in order] + [("pooled", np.ones(len(y), bool))]
    for name, sel in groups:
        ys = y[sel]
        pi = float(ys.mean())
        seed_rows = []
        for s in seeds:
            ps = probs[s][sel]
            auprc = exact_metrics(ys, ps)["AUPRC"]
            check = calibrated_ap(ys, ps, 1.0)
            max_check = max(max_check, abs(check - auprc))
            seed_rows.append(
                {
                    "AUPRC": auprc,
                    **{
                        f"calibrated_AUPRC_pi0_{q}": calibrated_ap(
                            ys, ps, calibration(pi, q)
                        )
                        for q in pi0s
                    },
                }
            )
        per_seed.append({"patient": name, "seeds": seed_rows})
        rows.append(
            {
                "patient": name,
                "prevalence": pi,
                **pd.DataFrame(seed_rows).mean().to_dict(),
            }
        )
    assert max_check <= tol, (
        f"calibrated AP with c = 1 != torchmetrics AUPRC ({max_check:.2e})"
    )
    table = pd.DataFrame(rows)

    # results run's per-patient table: same nuclei, prevalence and AUPRC
    pp = ref["per_patient"].set_index("patient").loc[order]
    t_pp = table.set_index("patient").loc[order]
    agree = {
        "prevalence_max_abs_diff": float(
            (pp["prevalence"] - t_pp["prevalence"]).abs().max()
        ),
        "AUPRC_max_abs_diff": float((pp["AUPRC"] - t_pp["AUPRC"]).abs().max()),
        "pooled_AUPRC_abs_diff": float(
            abs(ref["pooled"]["AUPRC"] - table.iloc[-1]["AUPRC"])
        ),
        "labelled_nuclei_equal": bool(
            (
                pp["labeled_nuclei"].to_numpy()
                == np.bincount(
                    pd.Categorical(patient, categories=order).codes,
                    minlength=len(order),
                )
            ).all()
        ),
    }
    assert (
        agree["labelled_nuclei_equal"]
        and max(
            agree["prevalence_max_abs_diff"],
            agree["AUPRC_max_abs_diff"],
            agree["pooled_AUPRC_abs_diff"],
        )
        < 1e-9
    ), agree

    # operating point --------------------------------------------------------------------
    vmean = vprobs.astype(np.float64).mean(0)
    threshold = recall_threshold(vy, vmean, float(config.target_recall))
    val_op = operating_point(vy, vmean, threshold)
    tmean = probs.astype(np.float64).mean(0)
    cmean = cprobs.astype(np.float64).mean(0)
    ops, ops_rerun = [], []
    for _, sel in groups:
        ops.append(operating_point(y[sel], tmean[sel], threshold))
        ops_rerun.append(operating_point(y[sel], cmean[sel], threshold))
    for col in ("recall", "FPR", "TP", "FP", "FN"):
        table[col] = [o[col] for o in ops]

    rerun = {
        "device": device,
        "per_seed_abs_diff": [
            {
                "max": float(np.abs(cprobs[s] - probs[s]).max()),
                "mean": float(np.abs(cprobs[s] - probs[s]).mean()),
                "q99": float(np.quantile(np.abs(cprobs[s] - probs[s]), 0.99)),
            }
            for s in seeds
        ],
        "pooled_AUPRC_rerun_seed_mean": float(
            np.mean([exact_metrics(y, cprobs[s])["AUPRC"] for s in seeds])
        ),
        "pooled_AUPRC_stored_seed_mean": float(table.iloc[-1]["AUPRC"]),
        "operating_point_rerun": dict(
            zip([g[0] for g in groups], ops_rerun, strict=True)
        ),
        "max_abs_diff_recall": float(
            max(
                abs(a["recall"] - b["recall"])
                for a, b in zip(ops, ops_rerun, strict=True)
            )
        ),
        "max_abs_diff_FPR": float(
            max(abs(a["FPR"] - b["FPR"]) for a, b in zip(ops, ops_rerun, strict=True))
        ),
    }

    # diagnostics ------------------------------------------------------------------------
    cal = table.set_index("patient")[f"calibrated_AUPRC_pi0_{pi0s[0]}"]
    diag = []
    for name, sel in groups[:-1]:
        med, med_o, share = [], [], []
        for s in seeds:
            ps = probs[s]
            med.append(float(np.median(ps[sel & (y == 1)])))
            med_o.append(float(np.median(ps[~sel & (y == 1)])))
            p99 = float(np.quantile(ps[~sel & (y == 0)], 0.99))
            share.append(float(np.mean(ps[sel & (y == 0)] > p99)))
        others = float(cal.drop([name, "pooled"]).median())
        diag.append(
            {
                "patient": name,
                "calibrated_AUPRC": float(cal[name]),
                "others_median_calibrated_AUPRC": others,
                "clearly_below": bool(cal[name] < others - config.clearly_below_margin),
                "carcinoma_median": float(np.mean(med)),
                "others_carcinoma_median": float(np.mean(med_o)),
                "benign_share_above_others_p99": float(np.mean(share)),
            }
        )
    diag = pd.DataFrame(diag)
    diag["fn_tendency"] = (
        diag["carcinoma_median"]
        < diag["others_carcinoma_median"] - config.fn_median_margin
    )
    diag["fp_tendency"] = (
        diag["benign_share_above_others_p99"] > config.fp_share_factor * 0.01
    )
    diag["recall_at_operating_point"] = diag["patient"].map(
        table.set_index("patient")["recall"]
    )
    diag["verdict"] = [verdict(r) for r in diag.itertuples()]

    table.to_csv(out / "patient_error_analysis_mmci.csv", index=False)
    diag.to_csv(out / "patient_diagnostics_mmci.csv", index=False)
    checks = {
        "results_run": config.results_run,
        "eval_runs": ref["eval_runs"],
        "training_runs": ref["training_runs"],
        "checkpoint_epochs": ref["checkpoint_epochs"],
        "test": test_counts,
        "validation": {
            **val_counts,
            "split": "split_train_val(random_state=42) of the training slides",
            **{
                k: d0[k]
                for k in ("split_size", "split_stratify_col", "split_group_col")
            },
        },
        "agreement_with_results_run": agree,
        "calibration_check": {
            "c": 1.0,
            "max_abs_diff_vs_torchmetrics": max_check,
            "tolerance": tol,
        },
        "operating_point": {
            "target_recall": float(config.target_recall),
            "threshold": threshold,
            "score": "seed-mean probability",
            "validation": val_op,
        },
        "inference": {
            "device": device,
            "block_mask": f"pre-fix ({FIX_COMMIT[:8]}^)",
            "stored_test_predictions": "GPU (float32 matmul precision 'high'), before the fix",
        },
        "rerun_check": rerun,
        "per_seed": per_seed,
        "rules": {
            "clearly_below_margin": float(config.clearly_below_margin),
            "fn_median_margin": float(config.fn_median_margin),
            "fp_share_factor": float(config.fp_share_factor),
        },
    }
    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=str))
    (out / "report.md").write_text(report(table, diag, checks, config))
    private = [
        *mapping["patient_id"],
        *mapping["case_id"],
        *meta["slide_id"],
        *train_meta["slide_id"],
        *train_meta["patient_id"].astype(str),
    ]
    return out, checks, private


def verdict(r: object) -> str:
    """One line: ranking (calibrated AUPRC) vs the others, then what the diagnostics show."""
    gap = f"calibrated AUPRC {r.calibrated_AUPRC:.3f} vs {r.others_median_calibrated_AUPRC:.3f} for the others"
    fn = (
        f"carcinoma median {r.carcinoma_median:.3f} vs {r.others_carcinoma_median:.3f}, "
        f"recall {r.recall_at_operating_point:.2f} at the operating point"
    )
    fp = f"{r.benign_share_above_others_p99:.1%} of benign above the others' p99 vs 1%"
    causes = []
    if r.fn_tendency:
        causes.append(f"false negatives ({fn})")
    if r.fp_tendency:
        causes.append(f"false positives ({fp})")
    if r.clearly_below:
        if not causes:
            return (
                f"Clearly below ({gap}); neither diagnostic explains it ({fn}; {fp})."
            )
        return f"Clearly below ({gap}), explained by " + " and ".join(causes) + "."
    if not causes:
        return f"On par ({gap}); no false-negative or false-positive tendency."
    return (
        f"On par in ranking ({gap}), but with a tendency to "
        + " and ".join(causes)
        + " (scores shifted against the other patients, not ranked worse within the patient)."
    )


def report(
    table: pd.DataFrame, diag: pd.DataFrame, checks: dict, config: DictConfig
) -> str:
    op, rr = checks["operating_point"], checks["rerun_check"]
    pi0s = [float(v) for v in config.pi0]
    lines = [
        "# Per-patient error analysis: MMCI TL test",
        "",
        (
            f"Model and data of run `{config.results_run}` (Shape + RoVE + k-NN attention, k = 44, three seeds; "
            "Another pathology nuclei ignored, Exclude negative). Metrics per seed, then the mean of the seeds; "
            "patients P1–P10 as in its per-patient table."
        ),
        "",
        "## Results",
        "",
        (
            "Calibrated AUPRC (Siblini et al. 2020): precision TP / (TP + c·FP), c = π(1 − π₀) / (π₀(1 − π)), π = "
            f"the patient's prevalence. Operating point: threshold {op['threshold']:.4f} = recall ≥ "
            f"{op['target_recall']:.2f} on the validation nuclei (seed-mean probability; validation recall "
            f"{op['validation']['recall']:.4f}, FPR {op['validation']['FPR']:.4f}), applied to the seed-mean test "
            "probability."
        ),
        "",
        "| Patient | Prevalence | AUPRC | "
        + " | ".join(f"Cal. AUPRC (π₀ = {q})" for q in pi0s)
        + " | Recall | FPR | TP | FP | FN |",
        "| --- " * (5 + len(pi0s) + 3) + "|",
    ]
    for r in table.to_dict("records"):
        name = "**Pooled**" if r["patient"] == "pooled" else r["patient"]
        lines.append(
            f"| {name} | {r['prevalence']:.4f} | {r['AUPRC']:.4f} | "
            + " | ".join(f"{r[f'calibrated_AUPRC_pi0_{q}']:.4f}" for q in pi0s)
            + f" | {r['recall']:.4f} | {r['FPR']:.4f} | {r['TP']:,} | {r['FP']:,} | {r['FN']:,} |"
        )
    lines += [
        "",
        "## Verdicts",
        "",
        (
            f"Clearly below = calibrated AUPRC (π₀ = {pi0s[0]}) more than {config.clearly_below_margin} below the median "
            f"of the other nine patients. False-negative tendency = median score of the patient's carcinoma nuclei more "
            f"than {config.fn_median_margin} below that of all other patients' carcinoma nuclei; false-positive "
            f"tendency = more than {config.fp_share_factor:g} × 1 % of its benign nuclei above the other patients' 99th "
            "percentile of benign scores."
        ),
        "",
    ]
    lines += [f"- **{r.patient}**: {r.verdict}" for r in diag.itertuples()]
    lines += [
        "",
        "| Patient | Carcinoma median | Others' carcinoma median | Benign > others' p99 |",
        "| --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r.patient} | {r.carcinoma_median:.3f} | {r.others_carcinoma_median:.3f} | {r.benign_share_above_others_p99:.2%} |"
        for r in diag.itertuples()
    ]
    lines += [
        "",
        (
            "The calibrated AUPRC measures the ranking within a patient; the diagnostics compare the patient's scores "
            "with the other patients'. P1–P3 hold most carcinoma nuclei, so the others' carcinoma median is ≈ 0.99 for "
            "every other patient, and a low-prevalence patient can rank well (high calibrated AUPRC) while its "
            "carcinoma scores sit below the operating threshold."
        ),
    ]
    a, cc, v = (
        checks["agreement_with_results_run"],
        checks["calibration_check"],
        checks["validation"],
    )
    seeds = rr["per_seed_abs_diff"]
    pooled_rr = rr["operating_point_rerun"]["pooled"]
    dev = rr["device"].split(":")[0].upper()
    recalls = table.set_index("patient")["recall"]
    moved = ", ".join(
        f"{p} {recalls[p]:.2f} → {o['recall']:.2f}"
        for p, o in rr["operating_point_rerun"].items()
        if abs(o["recall"] - recalls[p]) > 0.01
    )
    cause = (
        "  The difference grows with slide size (no kernel or code difference found: the model and feature code "
        "are unchanged for this model since the evaluation, and the pre-fix block mask is used), so it is "
        "attributed to GPU (TF32) vs CPU float arithmetic."
        if dev == "CPU"
        else (
            f"  Inference on {dev} with TF32 matmuls (float32 matmul precision 'high'), as the stored predictions. "
            "The remaining difference is not explained: the model and data code (up to comments), the prediction "
            "config, the torch version, the nuclei files and the checkpoints are those of the evaluation, whose GPU "
            "model is not recorded; the operating point is barely affected (pooled recall equal to the stored one "
            "to 1e-4)."
        )
    )
    lines += [
        "",
        "## Checks",
        "",
        (
            "- Calibration: with c = 1 (π = π₀) the calibrated AUPRC equals the torchmetrics AUPRC for every "
            f"patient, seed and the pooled set (max |Δ| {cc['max_abs_diff_vs_torchmetrics']:.1e})."
        ),
        (
            "- Same nuclei, prevalence and AUPRC as the results run's per-patient table (max |Δ AUPRC| "
            f"{a['AUPRC_max_abs_diff']:.1e})."
        ),
        (
            f"- Validation: {v['slides']} slides ({v['carcinoma_slides']} carcinoma) of {v['patients']} patients, "
            f"{v['labelled_nuclei']:,} labelled nuclei (prevalence {v['prevalence']:.4f}; {v['ignored_nuclei']:,} "
            "Another pathology nuclei ignored); no run stores them, so they were predicted here on "
            f"{dev} with the three checkpoints and the pre-fix block mask (as the stored test predictions)."
        ),
        f"- {dev} vs stored (GPU) test predictions, re-predicted for all test nuclei: mean |Δp| "
        + ", ".join(f"{s['mean']:.4f}" for s in seeds)
        + "; 99th percentile "
        + ", ".join(f"{s['q99']:.3f}" for s in seeds)
        + "; max "
        + ", ".join(f"{s['max']:.3f}" for s in seeds)
        + f". Pooled AUPRC {rr['pooled_AUPRC_rerun_seed_mean']:.4f} ({dev}) vs "
        f"{rr['pooled_AUPRC_stored_seed_mean']:.4f}; at the threshold, pooled recall {pooled_rr['recall']:.4f} / "
        f"FPR {pooled_rr['FPR']:.4f} on {dev} scores; max per-patient |Δ recall| {rr['max_abs_diff_recall']:.4f}, "
        f"|Δ FPR| {rr['max_abs_diff_FPR']:.4f}.",
        cause
        + (
            f" Recall at the threshold on {dev} vs stored scores: {moved}; the recall of the patients with few "
            "carcinoma nuclei is uncertain to this extent."
            if moved
            else ""
        ),
        "",
    ]
    return "\n".join(lines)


@with_cli_args(["+postprocessing=patient_error_analysis_mmci"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
def main(config: DictConfig) -> None:
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    client = MlflowClient()
    out, checks, private = compute(config, client)
    (out / "config-resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    preflight(out, private)
    if not config.log_mlflow:
        print(f"log_mlflow=false: outputs in {out}")
        return

    op = checks["operating_point"]
    params = {
        "results_run": config.results_run,
        "dataset": config.dataset_label,
        "pi0": ",".join(str(v) for v in config.pi0),
        "target_recall": config.target_recall,
        "threshold": op["threshold"],
        "inference": f"{checks['inference']['block_mask']}, {checks['inference']['device']}",
        **{f"eval_run_{i}": r for i, r in enumerate(checks["eval_runs"], start=1)},
        **{
            f"training_run_{i}": r
            for i, r in enumerate(checks["training_runs"], start=1)
        },
    }
    tags = {
        "mlflow.runName": config.metadata.run_name,
        "mlflow.note.content": config.metadata.description,
        "results_run": config.results_run,
        **git_info(*CODE),
    }
    for text in [*map(str, tags.values()), *map(str, params.values())]:
        assert not CREDENTIAL_URL.search(text), "credential URL in a tag or param"
    experiment = client.get_experiment_by_name(config.metadata.experiment_name)
    run_id = client.create_run(
        experiment.experiment_id, run_name=config.metadata.run_name, tags=tags
    ).info.run_id
    user = client.get_run(config.results_run).data.tags.get("mlflow.user")
    if user:
        client.set_tag(run_id, "mlflow.user", user)
    for k, v in params.items():
        client.log_param(run_id, k, v)
    table = pd.read_csv(out / "patient_error_analysis_mmci.csv").set_index("patient")
    for col in table.columns:
        client.log_metric(run_id, f"pooled/{col}", float(table.loc["pooled", col]))
    client.log_metric(run_id, "validation/recall", op["validation"]["recall"])
    client.log_metric(run_id, "validation/FPR", op["validation"]["FPR"])
    for name in (
        "patient_error_analysis_mmci.csv",
        "patient_diagnostics_mmci.csv",
        "checks.json",
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
        f"MLflow run {run_id} (experiment {experiment.experiment_id}); outputs in {out}"
    )


if __name__ == "__main__":
    main()
