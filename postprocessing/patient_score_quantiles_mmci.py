# ruff: noqa: RUF002  (Greek letters / minus signs in report text are intended)
r"""Per-patient score quantiles and operating points on the MMCI TL test set.

Model, nuclei, labels and P1-P10 of the per-patient error analysis (``parent_run``, d73e1fe0; itself
built on run dc419c48): Shape + RoVE + k-NN attention (k = 44), three seeds, Another pathology nuclei
ignored, scores = the seed-mean probability. Nothing is predicted here:

- test: the stored per-nucleus predictions of run dc419c48's evaluation runs (the plotted scores);
- validation: the GPU predictions (pre-fix block mask, TF32) that ``parent_run`` cached for the fixed
  validation split of the training slides; the run fails if the cache is missing.

Outputs (``out_dir``; logged to a child run of ``parent_run``):

1. ``patient_score_quantiles_mmci.csv``: one row per patient and class (carcinoma / benign): n,
   the patient's prevalence and the 5/25/50/75/95th percentiles of the seed-mean test score
   (``np.quantile``, linear interpolation).
2. ``thresholds.json``: the recall-0.90 threshold (highest score with validation recall ≥ 0.90; must
   equal ``parent_run``'s) and the Youden threshold (score t maximising TPR − FPR of s ≥ t over the
   distinct validation scores; the highest such t on ties), each with validation recall and FPR.
3. ``patient_operating_points_mmci.csv``: recall and FPR (with TP, FP, FN) per test patient and pooled
   at both thresholds.
"""

import json
import shutil
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import with_cli_args

from nuclei_graph.data.datamodules.base import split_train_val
from postprocessing.bootstrap_sources import load_run
from postprocessing.nucleus_results_mmci import CREDENTIAL_URL, preflight, run_config
from postprocessing.patient_error_analysis_mmci import (
    labelled,
    operating_point,
    recall_threshold,
    reference,
)
from postprocessing.tile.bootstrap_pixel_vs_nuclei import git_info


REPO = Path(__file__).parents[1]
CODE = (
    "postprocessing/patient_score_quantiles_mmci.py",
    "configs/postprocessing/patient_score_quantiles_mmci.yaml",
)


def youden_threshold(y: NDArray, s: NDArray) -> tuple[float, float]:
    """(t, J): the score t maximising J = TPR − FPR of s ≥ t over the distinct scores (highest t on ties)."""
    order = np.argsort(-s, kind="stable")
    ss, ys = s[order], y[order].astype(np.int64)
    last = np.r_[np.flatnonzero(ss[1:] != ss[:-1]), len(ss) - 1]
    tp, fp = np.cumsum(ys)[last], np.cumsum(1 - ys)[last]
    j = tp / tp[-1] - fp / fp[-1]
    i = int(np.argmax(j))
    return float(ss[last[i]]), float(j[i])


def compute(config: DictConfig, client: MlflowClient) -> tuple[Path, dict, list[str]]:
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_tmp"
    tmp.mkdir(exist_ok=True)

    parent = client.get_run(config.parent_run)
    assert parent.data.params["results_run"] == config.results_run
    ref = reference(config, client, tmp)
    for i, (e, t) in enumerate(
        zip(ref["eval_runs"], ref["training_runs"], strict=True), start=1
    ):
        assert parent.data.params[f"eval_run_{i}"] == e
        assert parent.data.params[f"training_run_{i}"] == t
    assert parent.data.params["inference"].endswith(f", {config.inference_device}")

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

    # validation: the parent run's cached predictions of the training runs' fixed split
    train_cfgs = [run_config(client, t, tmp) for t in ref["training_runs"]]
    d0 = train_cfgs[0]["data"]
    train_meta = pd.read_parquet(download_artifacts(d0["metadata"]["train"]))
    _, val = split_train_val(
        train_meta, d0["split_size"], d0["split_stratify_col"], d0["split_group_col"]
    )
    val = val.sort_values("slide_id", ignore_index=True)
    val["patient_id"] = val["patient_id"].astype(str)
    cache = Path(config.inference_cache_dir) / str(config.inference_device)
    paths = [cache / f"{t}_validation.parquet" for t in ref["training_runs"]]
    missing = [p.name for p in paths if not p.exists()]
    assert not missing, f"cached validation predictions missing: {missing}"
    val_pred = [pd.read_parquet(p) for p in paths]
    assert all(int(f["slide"].max()) == len(val) - 1 for f in val_pred)
    vy, _, vprobs, val_counts = labelled(config, val, val_pred)
    shutil.rmtree(tmp, ignore_errors=True)

    # thresholds on the seed-mean validation score (as the parent run)
    vmean = vprobs.astype(np.float64).mean(0)
    t90 = recall_threshold(vy, vmean, float(config.target_recall))
    assert t90 == float(parent.data.params["threshold"]), (
        f"recall threshold {t90} != parent run's {parent.data.params['threshold']}"
    )
    t_youden, j = youden_threshold(vy, vmean)
    thresholds = {
        "recall_0.90": {
            "threshold": t90,
            "rule": f"highest score with validation recall >= {config.target_recall}",
            "validation": operating_point(vy, vmean, t90),
        },
        "youden": {
            "threshold": t_youden,
            "rule": "argmax over distinct validation scores t of TPR - FPR (s >= t); highest t on ties",
            "J": j,
            "validation": operating_point(vy, vmean, t_youden),
        },
    }

    # per patient
    mapping = pd.read_csv(config.patient_label_mapping, dtype=str)
    plabel = dict(zip(mapping["patient_id"], mapping["patient"], strict=True))
    assert set(plabel) == set(meta["patient_id"])
    patient = meta["patient_id"].map(plabel).to_numpy()[slide]
    order = sorted(plabel.values(), key=lambda p: int(p[1:]))
    tmean = probs.astype(np.float64).mean(0)
    qs = [float(q) for q in config.quantiles]

    rows = []
    for p in order:
        sel = patient == p
        prevalence = float(y[sel].mean())
        for cls, label in (("carcinoma", 1), ("benign", 0)):
            s = tmean[sel & (y == label)]
            rows.append(
                {
                    "patient": p,
                    "class": cls,
                    "n": len(s),
                    "prevalence": prevalence,
                    **{
                        f"p{round(q * 100)}": float(v)
                        for q, v in zip(qs, np.quantile(s, qs), strict=True)
                    },
                }
            )
    quantiles = pd.DataFrame(rows)
    assert quantiles["n"].sum() == len(y)

    # the parent's per-patient recall at the recall-0.90 threshold (same scores)
    parent_table = pd.read_csv(
        client.download_artifacts(
            config.parent_run, "patient_error_analysis_mmci.csv", str(out)
        )
    ).set_index("patient")
    (out / "patient_error_analysis_mmci.csv").unlink()

    ops = []
    groups = [(p, patient == p) for p in order] + [("pooled", np.ones(len(y), bool))]
    for name, sel in groups:
        row: dict[str, object] = {"patient": name}
        for key, t in (("recall90", t90), ("youden", t_youden)):
            o = operating_point(y[sel], tmean[sel], t)
            row |= {f"{k}_{key}": v for k, v in o.items()}
        assert abs(row["recall_recall90"] - parent_table.loc[name, "recall"]) < 1e-12
        assert abs(row["FPR_recall90"] - parent_table.loc[name, "FPR"]) < 1e-12
        ops.append(row)
    ops = pd.DataFrame(ops)

    quantiles.to_csv(out / "patient_score_quantiles_mmci.csv", index=False)
    ops.to_csv(out / "patient_operating_points_mmci.csv", index=False)
    checks = {
        "parent_run": config.parent_run,
        "results_run": config.results_run,
        "eval_runs": ref["eval_runs"],
        "training_runs": ref["training_runs"],
        "score": "seed-mean probability (3 seeds), Another pathology nuclei ignored",
        "test_scores": "stored predictions of the evaluation runs (GPU, pre-fix block mask)",
        "validation_scores": f"predictions cached by the parent run ({config.inference_device}, pre-fix block mask, TF32)",
        "quantiles": "np.quantile, linear interpolation",
        "thresholds": thresholds,
        "test": test_counts,
        "validation": val_counts,
        "checks": {
            "recall_0.90_threshold_equals_parent": True,
            "per_patient_recall_FPR_at_recall_0.90_equal_parent": True,
        },
    }
    (out / "thresholds.json").write_text(json.dumps(checks, indent=1, default=str))
    private = [
        *mapping["patient_id"],
        *mapping["case_id"],
        *meta["slide_id"],
        *train_meta["slide_id"],
        *train_meta["patient_id"].astype(str),
    ]
    return out, checks, private


@with_cli_args(["+postprocessing=patient_score_quantiles_mmci"])
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

    th = checks["thresholds"]
    params = {
        "parent_run": config.parent_run,
        "results_run": config.results_run,
        "dataset": config.dataset_label,
        "target_recall": config.target_recall,
        "threshold_recall_0.90": th["recall_0.90"]["threshold"],
        "threshold_youden": th["youden"]["threshold"],
        "quantiles": ",".join(str(q) for q in config.quantiles),
        "validation_scores": f"cached by parent run ({config.inference_device})",
    }
    parent = client.get_run(config.parent_run)
    tags = {
        "mlflow.runName": config.metadata.run_name,
        "mlflow.note.content": config.metadata.description,
        "mlflow.parentRunId": config.parent_run,
        "results_run": config.results_run,
        **git_info(*CODE),
    }
    for text in [*map(str, tags.values()), *map(str, params.values())]:
        assert not CREDENTIAL_URL.search(text), "credential URL in a tag or param"
    run_id = client.create_run(
        parent.info.experiment_id, run_name=config.metadata.run_name, tags=tags
    ).info.run_id
    user = parent.data.tags.get("mlflow.user")
    if user:
        client.set_tag(run_id, "mlflow.user", user)
    for k, v in params.items():
        client.log_param(run_id, k, v)
    for key, name in (("recall_0.90", "recall90"), ("youden", "youden")):
        client.log_metric(run_id, f"threshold/{name}", th[key]["threshold"])
        client.log_metric(
            run_id, f"validation/recall_{name}", th[key]["validation"]["recall"]
        )
        client.log_metric(
            run_id, f"validation/FPR_{name}", th[key]["validation"]["FPR"]
        )
    client.log_metric(run_id, "validation/youden_J", th["youden"]["J"])
    ops = pd.read_csv(out / "patient_operating_points_mmci.csv").set_index("patient")
    for col in ("recall_recall90", "FPR_recall90", "recall_youden", "FPR_youden"):
        client.log_metric(run_id, f"pooled/{col}", float(ops.loc["pooled", col]))
    for name in (
        "patient_score_quantiles_mmci.csv",
        "patient_operating_points_mmci.csv",
        "thresholds.json",
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
    assert r.data.tags["mlflow.parentRunId"] == config.parent_run
    client.set_terminated(run_id)
    print(f"MLflow run {run_id} (child of {config.parent_run}); outputs in {out}")


if __name__ == "__main__":
    main()
