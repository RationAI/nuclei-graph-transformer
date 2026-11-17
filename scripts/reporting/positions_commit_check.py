"""Do stored test predictions of an evaluation run at another commit reproduce at the reference commit?

For the positions ablation (tab:res-positions). The thesis tables use evaluation runs made at
commit 96860cec; an evaluation run made at another commit is checked as the blank-token seed 1
was (1f5e2e4a, commit a2f10781): its checkpoint re-predicts the same 24-slide test sample
(`_tmp_commit_check/96860cec_testsample_slides.parquet`, 21 carcinoma + 3 benign slides) with the
model code of the reference commit (git worktree, `postprocessing/nucleus_inference_at_commit.py`),
and, as the noise floor of re-prediction (TF32 matmuls, nondeterministic kernels), with the code of
the run's own commit. Both are compared with the run's stored predictions on those slides.

Inputs come from `scripts/reporting/positions_ablation_preflight.py` (no MLflow needed here): the
evaluation run's resolved config and the training run's checkpoint under `_tmp_commit_check/`,
the stored predictions in the prediction cache. Paths of the public cluster's project folder in
the config are remapped to the read-only mount here.

Writes `<out_dir>/commit_check.csv` (run ids and numbers only, no slide ids).

Usage:
    uv run python scripts/reporting/positions_commit_check.py --eval-run 8eef7874... [--eval-run ...]
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO))

from postprocessing.bootstrap_sources import (
    exact_metrics,
    label_predictions,
    load_labels,
)


REMOTE = "/mnt/projects/nuclei_based_wsi_analysis"
PUBLIC = "/mnt/public_projects/nuclei_based_wsi_analysis"
RADBOUD_TEST = Path(REMOTE) / "evaluation/radboud_test"
CHECK_DIR = RADBOUD_TEST / "_tmp_commit_check"
SAMPLE = CHECK_DIR / "96860cec_testsample_slides.parquet"
CACHE = RADBOUD_TEST / "predictions"
INFERENCE_OUT = RADBOUD_TEST / "inference_predictions"
OUT_DIR = RADBOUD_TEST / "positions_ablation"
RESAMPLES = (
    Path(PUBLIC)
    / "evaluation/radboud_test/q2_sources_bootstrap/bootstrap_resamples_radboud_test.npz"
)
SUPERVISION = Path(PUBLIC) / "nuclei_supervision/PANDA/Radboud/annotation"
REFERENCE_COMMIT = "96860cec"
INFERENCE = "postprocessing/nucleus_inference_at_commit.py"


def remap(obj):
    if isinstance(obj, dict):
        return {k: remap(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [remap(v) for v in obj]
    if isinstance(obj, str) and obj.startswith(REMOTE):
        return PUBLIC + obj[len(REMOTE) :]
    return obj


def predict(
    commit: str, eval_run: str, train_run: str, ckpt: Path, cfg_path: Path
) -> Path:
    out = INFERENCE_OUT / commit[:8] / f"{train_run}_testsample.parquet"
    if out.exists():
        return out
    job = CHECK_DIR / f"{commit[:8]}_testsample_{train_run[:8]}_job.json"
    job.write_text(
        json.dumps(
            {
                "slides": str(SAMPLE),
                "runs": [{"checkpoint": str(ckpt), "config": str(cfg_path)}],
                "outputs": [str(out)],
                "device": "cuda",
            }
        )
    )
    worktree = CHECK_DIR / f"code_{commit[:8]}"
    if not worktree.exists():
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), commit],
            cwd=REPO,
            check=True,
            capture_output=True,
        )
    try:
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
            env={**os.environ, "PYTHONPATH": str(worktree)},
            check=True,
        )
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=REPO,
            check=False,
        )
    return out


def compare(stored: pd.DataFrame, new: pd.DataFrame, y: np.ndarray) -> dict:
    assert np.array_equal(stored["slide"].to_numpy(), new["slide"].to_numpy())
    assert np.array_equal(stored["key"].to_numpy(), new["key"].to_numpy()), (
        "different nuclei"
    )
    a, b = stored["p"].to_numpy(np.float64), new["p"].to_numpy(np.float64)
    d = np.abs(a - b)
    ea, eb = exact_metrics(y, a), exact_metrics(y, b)
    return {
        "nuclei": len(a),
        "max_abs_p_diff": float(d.max()),
        "mean_abs_p_diff": float(d.mean()),
        "p99_abs_p_diff": float(np.quantile(d, 0.99)),
        "pearson_p": float(np.corrcoef(a, b)[0, 1]),
        "same_side_of_0.5": float(np.mean((a >= 0.5) == (b >= 0.5))),
        "AUPRC_stored": ea["AUPRC"],
        "AUPRC_repredicted": eb["AUPRC"],
        "AUROC_stored": ea["AUROC"],
        "AUROC_repredicted": eb["AUROC"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-run", action="append", required=True)
    args = parser.parse_args()

    runs = pd.read_csv(OUT_DIR / "preflight_runs.csv")
    test_ids = list(
        np.load(RESAMPLES)["slide_ids"]
    )  # sorted test slides = cache slide index
    sample = pd.read_parquet(SAMPLE)
    sample_ids = sample["slide_id"].tolist()
    to_test = np.array([test_ids.index(s) for s in sample_ids])
    labels = load_labels(sample, SUPERVISION, "annot_label")

    rows = []
    for eval_run in args.eval_run:
        r = runs[runs["eval_run"].str.startswith(eval_run)].iloc[0]
        cfg = yaml.safe_load(
            (CHECK_DIR / r.eval_run / "configs/config-resolved.yaml").read_text()
        )
        config_source = "evaluation run"
        if (
            "model" not in cfg
        ):  # not recorded (a later metrics step overwrote it): training's
            cfg = yaml.safe_load(
                (
                    CHECK_DIR / r.training_run / "configs/config-resolved.yaml"
                ).read_text()
            )
            config_source = "training run (evaluation config not recorded)"
        cfg = remap(cfg)
        cfg["data"].setdefault("context", "knn")  # configs before the option: k-NN
        cfg_path = CHECK_DIR / r.eval_run / "config-remapped.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        ckpt = CHECK_DIR / r.local_checkpoint
        stored = pd.read_parquet(CACHE / f"{r.eval_run}.parquet")
        stored = stored[stored["slide"].isin(to_test)].copy()
        stored["slide"] = (
            pd.Series(range(len(to_test)), index=to_test)
            .loc[stored["slide"]]
            .to_numpy()
        )
        stored = stored.astype({"slide": np.int16}).sort_values(
            ["slide", "key"], ignore_index=True
        )
        y, _ = label_predictions(stored, labels, sample["is_carcinoma"].to_numpy(bool))
        for commit, role in (
            (REFERENCE_COMMIT, "reference"),
            (str(r.eval_commit), "own (noise floor)"),
        ):
            try:
                new = pd.read_parquet(
                    predict(commit, r.eval_run, r.training_run, ckpt, cfg_path)
                )
            except subprocess.CalledProcessError as e:
                if role == "reference":
                    raise
                print(f"{r.eval_run[:8]} at {commit[:8]}: failed ({e})", flush=True)
                rows.append(
                    {
                        "configuration": r.configuration,
                        "eval_run": r.eval_run,
                        "repredicted_at": commit[:8],
                        "role": f"{role}: inference script does not run at this commit",
                    }
                )
                continue
            rows.append(
                {
                    "config_source": config_source,
                    "configuration": r.configuration,
                    "seed": r.seed,
                    "eval_run": r.eval_run,
                    "training_run": r.training_run,
                    "eval_commit": r.eval_commit,
                    "repredicted_at": commit[:8],
                    "role": role,
                    "sample_slides": len(sample_ids),
                    **compare(stored, new, y),
                }
            )
            print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    text = out.to_csv(index=False)
    assert not any(s in text for s in test_ids)
    (OUT_DIR / "commit_check.csv").write_text(text)


if __name__ == "__main__":
    main()
