"""Final representations h of the context configurations, for tab:res-context-probes.

For the self-only, k-NN (no position encoding) and dense models of tab:res-context (three seeds
each), `h` = the output of `net.final_norm` with each evaluation run's own code and mask: the
checkpoint runs in a git worktree of the evaluation run's commit
(`postprocessing/nucleus_inference_at_commit.py`, whole slides, TF32), with the evaluation run's
resolved config (paths remapped to the read-only mount). Nuclei are those of the analysed model's
XAI index (c8f26853):
- validation: a fixed uniform random sample of 2,000,000 nuclei (seed 20261119), checked to put
  no slide above 1 % of the sample (the probes are fitted on it);
- test: every nucleus (the probes are evaluated on all of them).
Every nucleus of a split is predicted; the test probabilities are compared with the stored
predictions of the evaluation run (`check`).

Self-only seed 1 (24ef0778): its evaluation config was overwritten by a metrics step; its
prediction config at commit 99510237 (configs/experiment/modeling/prediction/nuclei_level/
radboud.yaml) is the training config with `context: none`, used here with the training run's
resolved config. Self-only seeds 2-3: see CONTEXT_OVERRIDE.

Subcommands (in order):
    prepare    checkpoints and configs from MLflow (needs the public MLflow credentials in the env)
    sample     slide tables per split and the validation sample (`--n`, `--seed`)
    test_rows  every test nucleus, in the order of the outputs
    pilot      24-slide test sample: variants of runs (see `PILOT`), compared with their predictions
    extract    all models x splits (resumable: an existing output is kept)
    check      stored vs re-predicted test probabilities, all test nuclei -> check.csv

Work folder: WORK (no slide ids in the CSV/JSON outputs; the slide tables keep them, as inputs).
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
    nucleus_keys,
)


REMOTE = "/mnt/projects/nuclei_based_wsi_analysis"
PUBLIC = "/mnt/public_projects/nuclei_based_wsi_analysis"
XAI = Path(PUBLIC) / "xai/c8f268536be8416f9fe395eea95c458e"
WORK = Path(REMOTE) / "evaluation/radboud_test/context_probes"
CACHE = Path(REMOTE) / "evaluation/radboud_test/predictions"
SAMPLE_24 = (
    Path(REMOTE)
    / "evaluation/radboud_test/_tmp_commit_check/96860cec_testsample_slides.parquet"
)
SUPERVISION = Path(PUBLIC) / "nuclei_supervision/PANDA/Radboud/annotation"
INFERENCE = REPO / "postprocessing/nucleus_inference_at_commit.py"
MAX_SLIDE_SHARE = 0.01  # of the validation sample
# (split, rows file, output tag)
PARTS = (
    ("validation", "sample_validation.parquet", "validation"),
    ("test", "test_rows.parquet", "test_all"),
)

# (configuration, seed, evaluation run, evaluation commit, config source)
MODELS = (
    ("self", 1, "24ef0778c482421ea1776caa0051bf35", "99510237", "training"),
    ("self", 2, "683bc01ddab944919dbc1f53317da338", "96860cec", "evaluation"),
    ("self", 3, "eaa33dc4279f4c53a001a4754ef144d5", "96860cec", "evaluation"),
    ("knn", 1, "28cfe9796d924a3ca31a3922947aca05", "96860cec", "evaluation"),
    ("knn", 2, "be5e23d0adad4c0b8f1e77da8057a7ba", "96860cec", "evaluation"),
    ("knn", 3, "8eef7874b48c4051aaa86b7b6be4c50f", "a2f10781", "evaluation"),
    ("dense", 1, "9363c68f37da442f89c940d80efa49d5", "b4379405", "evaluation"),
    ("dense", 2, "df298e11a4c543dbaf57a378baab98e6", "b4379405", "evaluation"),
    ("dense", 3, "ba58d016ece74741a93b409d49526e5a", "b4379405", "evaluation"),
)
# Self-only seeds 2-3: their logged evaluation configs say context knn (data and net), but their
# stored predictions are reproduced only with context none, as trained (pilot on the 24-slide
# test sample, pilot.csv: mean |dp| 6e-5 with none; Pearson 0.38 and AUPRC 0.49 vs 0.66 with knn).
CONTEXT_OVERRIDE = {
    "683bc01ddab944919dbc1f53317da338": "none",
    "eaa33dc4279f4c53a001a4754ef144d5": "none",
}
# Self-only seed 1's checkpoint as in its prediction config at 99510237
SELF1_CHECKPOINT = "checkpoints/epoch=293-step=64092/checkpoint.ckpt"


# --------------------------------------------------------------------------
# prepare
# --------------------------------------------------------------------------


def remap(obj):
    if isinstance(obj, dict):
        return {k: remap(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [remap(v) for v in obj]
    if isinstance(obj, str) and obj.startswith(REMOTE):
        return PUBLIC + obj[len(REMOTE) :]
    return obj


def with_context(cfg: dict, context: str) -> dict:
    """The config with the data and net attention context set to `context`."""
    cfg = json.loads(json.dumps(cfg))
    cfg["data"]["context"] = context
    cfg["model"]["net"]["config"]["context"] = context
    return cfg


def prepare() -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    rows = []
    for name, seed, eval_run, commit, source in MODELS:
        out = WORK / "models" / eval_run
        out.mkdir(parents=True, exist_ok=True)
        run = client.get_run(eval_run)
        train_run = run.data.tags["mlflow.parentRunId"]
        cfg_run = eval_run if source == "evaluation" else train_run
        cfg_path = mlflow.artifacts.download_artifacts(
            run_id=cfg_run,
            artifact_path="configs/config-resolved.yaml",
            dst_path=str(out),
        )
        cfg = yaml.safe_load(Path(cfg_path).read_text())
        if source == "evaluation":
            uri = cfg["checkpoint"]
            prefix = f"mlflow-artifacts:/37/{train_run}/artifacts/"
            assert uri.startswith(prefix), uri
            artifact = uri[len(prefix) :]
        else:
            artifact = SELF1_CHECKPOINT
        ckpt = out / "checkpoint.ckpt"
        if not ckpt.exists():
            path = mlflow.artifacts.download_artifacts(
                run_id=train_run,
                artifact_path=artifact,
                dst_path=str(out / "_download"),
            )
            Path(path).rename(ckpt)
        (out / "config.yaml").write_text(yaml.safe_dump(remap(cfg)))
        rows.append(
            {
                "configuration": name,
                "seed": seed,
                "eval_run": eval_run,
                "training_run": train_run,
                "commit": commit,
                "config_source": source,
                "checkpoint_artifact": artifact,
                "data_context": cfg["data"]["context"],
                "net_context": cfg["model"]["net"]["config"].get("context"),
                "k": cfg["data"]["k"],
                "block_size": cfg["data"]["block_size"],
                "attention_variant": cfg["model"]["net"]["config"].get(
                    "attention_variant"
                ),
                "use_rope": cfg["model"]["net"]["config"].get("use_rope"),
            }
        )
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(WORK / "models.csv", index=False)


# --------------------------------------------------------------------------
# nuclei
# --------------------------------------------------------------------------


def slide_tables() -> dict[str, pd.DataFrame]:
    slides = pd.read_parquet(XAI / "slides.parquet")
    slides["slide_nuclei_path"] = slides["slide_nuclei_path"].str.replace(
        REMOTE, PUBLIC, regex=False
    )
    return {
        split: slides[slides["split"] == split].sort_values(
            "slide_id", ignore_index=True
        )
        for split in ("validation", "test")
    }


def index_length(split: str) -> int:
    import pyarrow.parquet as pq

    return pq.ParquetFile(XAI / f"activations/{split}/index.parquet").metadata.num_rows


def index_rows(split: str, part: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """(slide, key, xai_row) of the XAI index rows `rows`, sorted by (slide, key)."""
    index = pd.read_parquet(
        XAI / f"activations/{split}/index.parquet", columns=["slide_id", "nucleus_id"]
    )
    chosen = index.iloc[rows]
    code = pd.Categorical(
        chosen["slide_id"].astype(str), categories=part["slide_id"]
    ).codes
    assert (code >= 0).all()
    df = pd.DataFrame(
        {
            "slide": code.astype(np.int16),
            "key": nucleus_keys(chosen["nucleus_id"]),
            "xai_row": rows,
        }
    ).sort_values(["slide", "key"], ignore_index=True)
    assert not df.duplicated(["slide", "key"]).any()
    return df


def validation_sample_check(df: pd.DataFrame) -> dict:
    per_slide = df["slide"].value_counts()
    share = float(per_slide.max() / len(df))
    assert share <= MAX_SLIDE_SHARE, f"a slide holds {share:.2%} of the sample"
    return {
        "sampled": len(df),
        "slides_in_sample": len(per_slide),
        "max_slide_share": share,
    }


def sample(n: int, seed: int) -> None:
    """Slide tables per split (sorted by slide id) and n random validation nuclei."""
    tables = slide_tables()
    for split, part in tables.items():
        part.to_parquet(WORK / f"slides_{split}.parquet", index=False)
    rng = np.random.default_rng(seed)
    n_index = index_length("validation")
    rows = np.sort(rng.choice(n_index, size=min(n, n_index), replace=False))
    df = index_rows("validation", tables["validation"], rows)
    info = {
        "seed": seed,
        "validation": {
            "slides": len(tables["validation"]),
            "nuclei": n_index,
            **validation_sample_check(df),
        },
    }
    path = WORK / "sample_validation.parquet"
    if path.exists():  # an existing sample must be this draw
        old = pd.read_parquet(path)
        assert old.equals(df), "the existing validation sample is not this draw"
    else:
        df.to_parquet(path, index=False)
    (WORK / "sample.json").write_text(json.dumps(info, indent=1))
    print(info)


def test_rows() -> None:
    """Every test nucleus of the XAI index, sorted by (slide, key)."""
    part = slide_tables()["test"]
    assert part.equals(pd.read_parquet(WORK / "slides_test.parquet"))
    n_index = index_length("test")
    df = index_rows("test", part, np.arange(n_index))
    df.to_parquet(WORK / "test_rows.parquet", index=False)
    print({"test": {"slides": len(part), "nuclei": len(df)}})


# --------------------------------------------------------------------------
# inference at a commit
# --------------------------------------------------------------------------


def run_job(commit: str, job: dict, name: str) -> None:
    job_path = WORK / "jobs" / f"{name}.json"
    job_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(json.dumps(job))
    worktree = WORK / "code" / f"code_{commit}_{name}"
    if worktree.exists():
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=REPO,
            check=False,
        )
    worktree.parent.mkdir(parents=True, exist_ok=True)
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
                str(INFERENCE),
                "--job",
                str(job_path),
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


def compare(stored: pd.DataFrame, new: pd.DataFrame, y: np.ndarray) -> dict:
    m = stored.merge(
        new, on=["slide", "key"], suffixes=("_stored", "_new"), validate="1:1"
    )
    assert len(m) == len(stored) == len(new), "different nuclei"
    d = np.abs(m["p_stored"].to_numpy(np.float64) - m["p_new"].to_numpy(np.float64))
    stored_m = exact_metrics(y, m["p_stored"].to_numpy())
    new_m = exact_metrics(y, m["p_new"].to_numpy())
    return {
        "nuclei": len(m),
        "max_abs_p_diff": float(d.max()),
        "mean_abs_p_diff": float(d.mean()),
        "p99_abs_p_diff": float(np.quantile(d, 0.99)),
        "pearson_p": float(np.corrcoef(m["p_stored"], m["p_new"])[0, 1]),
        **{f"{k}_stored": v for k, v in stored_m.items()},
        **{f"{k}_repredicted": v for k, v in new_m.items()},
    }


def stored_on(slides: pd.DataFrame, eval_run: str) -> tuple[pd.DataFrame, np.ndarray]:
    """The cached stored predictions of `eval_run` on `slides` (slide = row in `slides`), labels."""
    test = slide_tables()["test"]
    stored = pd.read_parquet(CACHE / f"{eval_run}.parquet")
    position = {s: i for i, s in enumerate(slides["slide_id"])}
    stored["slide"] = test["slide_id"].to_numpy()[stored["slide"].to_numpy()]
    stored = stored[stored["slide"].isin(position)]
    stored["slide"] = stored["slide"].map(position).astype(np.int16)
    stored = stored.sort_values(["slide", "key"], ignore_index=True)
    labels = load_labels(
        slides[["slide_id", "is_carcinoma"]], SUPERVISION, "annot_label"
    )
    y, _ = label_predictions(stored, labels, slides["is_carcinoma"].to_numpy(bool))
    return stored, y


# Variants of the pilot: (name, evaluation run, commit, context override or None)
PILOT = (
    ("self2_as_evaluated", "683bc01ddab944919dbc1f53317da338", "96860cec", None),
    ("self2_context_none", "683bc01ddab944919dbc1f53317da338", "96860cec", "none"),
    ("self1", "24ef0778c482421ea1776caa0051bf35", "99510237", None),
    ("knn1", "28cfe9796d924a3ca31a3922947aca05", "96860cec", None),
    ("dense1", "9363c68f37da442f89c940d80efa49d5", "b4379405", None),
)


def pilot() -> None:
    slides = pd.read_parquet(SAMPLE_24).reset_index(drop=True)
    rows = []
    for name, eval_run, commit, context in PILOT:
        model_dir = WORK / "models" / eval_run
        cfg = yaml.safe_load((model_dir / "config.yaml").read_text())
        if context is not None:
            cfg = with_context(cfg, context)
        cfg_path = WORK / "pilot" / f"{name}_config.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(cfg))
        out = WORK / "pilot" / f"{name}.parquet"
        if not out.exists():
            run_job(
                commit,
                {
                    "slides": str(SAMPLE_24),
                    "runs": [
                        {
                            "checkpoint": str(model_dir / "checkpoint.ckpt"),
                            "config": str(cfg_path),
                        }
                    ],
                    "outputs": [str(out)],
                    "device": "cuda",
                },
                f"pilot_{name}",
            )
        stored, y = stored_on(slides, eval_run)
        rows.append(
            {
                "variant": name,
                "eval_run": eval_run,
                "commit": commit,
                "data_context": cfg["data"]["context"],
                "net_context": cfg["model"]["net"]["config"].get("context"),
                **compare(stored, pd.read_parquet(out), y),
            }
        )
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(WORK / "pilot.csv", index=False)


def extract(only: list[str] | None) -> None:
    for name, seed, eval_run, commit, _ in MODELS:
        if only and name not in only:
            continue
        model_dir = WORK / "models" / eval_run
        config_path = model_dir / "config.yaml"
        if eval_run in CONTEXT_OVERRIDE:
            cfg = yaml.safe_load(config_path.read_text())
            config_path = model_dir / "config_extract.yaml"
            config_path.write_text(
                yaml.safe_dump(with_context(cfg, CONTEXT_OVERRIDE[eval_run]))
            )
        for split, rows_file, tag in PARTS:
            out = WORK / "outputs" / f"{eval_run}_{tag}.parquet"
            h_out = WORK / "outputs" / f"{eval_run}_{tag}_h.npy"
            if out.exists() and h_out.exists() and h_out.with_suffix(".json").exists():
                print(f"{name}{seed} {tag}: done, skipped", flush=True)
                continue
            run_job(
                commit,
                {
                    "slides": str(WORK / f"slides_{split}.parquet"),
                    "runs": [
                        {
                            "checkpoint": str(model_dir / "checkpoint.ckpt"),
                            "config": str(config_path),
                        }
                    ],
                    "outputs": [str(out)],
                    "h_sample": str(WORK / rows_file),
                    "h_outputs": [str(h_out)],
                    "device": "cuda",
                },
                f"{name}{seed}_{tag}",
            )


def check() -> None:
    slides = pd.read_parquet(WORK / "slides_test.parquet")
    rows = []
    for name, seed, eval_run, commit, _ in MODELS:
        out = WORK / "outputs" / f"{eval_run}_test_all.parquet"
        if not out.exists():
            continue
        stored, y = stored_on(slides, eval_run)
        h_info = {
            tag: json.loads((WORK / "outputs" / f"{eval_run}_{tag}_h.json").read_text())
            for _, _, tag in PARTS
        }
        rows.append(
            {
                "configuration": name,
                "seed": seed,
                "eval_run": eval_run,
                "commit": commit,
                "context_used": CONTEXT_OVERRIDE.get(eval_run, "as evaluation config"),
                **compare(stored, pd.read_parquet(out), y),
                **{
                    f"class_head_max_abs_diff_{t}": h_info[t]["class_head_max_abs_diff"]
                    for t in h_info
                },
                **{f"h_rows_{t}": h_info[t]["rows"] for t in h_info},
            }
        )
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(WORK / "check.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=["prepare", "sample", "test_rows", "pilot", "extract", "check"],
    )
    parser.add_argument("--n", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    if args.command == "prepare":
        prepare()
    elif args.command == "sample":
        sample(args.n, args.seed)
    elif args.command == "test_rows":
        test_rows()
    elif args.command == "pilot":
        pilot()
    elif args.command == "extract":
        extract(args.only)
    else:
        check()


if __name__ == "__main__":
    main()
