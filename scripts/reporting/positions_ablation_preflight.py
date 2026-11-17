"""Preflight of the positions ablation (tab:res-positions): runs, configs, commits, downloads.

PANDA Radboud test set, four configurations of the shape model (k-NN attention, k = 24), three
seeds each (NB-A-68): no position encoding, RoPE, rotated values (the thesis model) and
relative-position values. For every evaluation run (experiment 37) this resolves the full run ids
from the 8-character prefixes, and records: the training run (``mlflow.parentRunId``), the model
configuration of both (resolved configs), the test metadata, the checkpoint the evaluation loaded vs
the training run's best validation epoch, the evaluation commit, dates and statuses.

Downloads, for the later steps (nothing is computed here):
- the test predictions of every run into ``cache_dir`` (``postprocessing.bootstrap_sources.load_run``;
  runs already cached are not downloaded again);
- each training run's checkpoint and resolved config, and each evaluation run's resolved config,
  into ``commit_check_dir/<run>/``, for re-predicting a test sample at the reference commit.

Writes ``<out_dir>/preflight_runs.csv`` (run ids, no slide ids) and prints a summary. Needs the
public MLflow (experiment 37): set ``MLFLOW_TRACKING_URI`` and the credentials in the environment.

Usage:
    uv run python scripts/reporting/positions_ablation_preflight.py
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd
import yaml
from mlflow.artifacts import download_artifacts
from mlflow.tracking import MlflowClient


REPO = Path(__file__).parents[2]
sys.path.insert(0, str(REPO))

from postprocessing.bootstrap_sources import load_run
from postprocessing.results_sources import (
    checkpoint_check,
    training_schedule_check,
)


EXPERIMENT = "37"
PROJECT = Path("/mnt/projects/nuclei_based_wsi_analysis")
OUT_DIR = PROJECT / "evaluation/radboud_test/positions_ablation"
CACHE_DIR = PROJECT / "evaluation/radboud_test/predictions"
COMMIT_CHECK_DIR = PROJECT / "evaluation/radboud_test/_tmp_commit_check"
METADATA_URI = "mlflow-artifacts:/37/5b9c6f096026425fa2b04c4e74b7a6f3/artifacts/panda/slides_mapping_test.parquet"
REFERENCE_COMMIT = (
    "96860cec"  # evaluation commit of the other thesis tables (tab:res-sources)
)
BLOCK_MASK_FIX_DATE = "2026-10-30"
COMMIT_TAGS = ("mlflow.source.git.commit", "code_commit", "git.commit", "git_commit")
CREDENTIAL_URL = re.compile(r"https?://[^/\s\"']*@")

# configuration -> [(training prefix, evaluation prefix)], seeds in NB-A-68 order
RUNS = {
    "shape_knn_nopos": [
        ("b825727e", "28cfe979"),
        ("5de0289b", "be5e23d0"),
        ("650b1d8d", "8eef7874"),
    ],
    "shape_knn_rope": [
        ("c5fe3bbb", "76fc240c"),
        ("4e7eb9f2", "d4b47030"),
        ("edec8d28", "ba19f8df"),
    ],
    "shape_knn": [
        ("5401eebe", "a5ee0cd9"),
        ("084da1a0", "98fc505c"),
        ("c8f26853", "777fb413"),
    ],
    "shape_knn_relpos": [
        ("090ac786", "4dda120e"),
        ("f886770", "e72e2d6e"),
        ("d2205401", "27960cda"),
    ],
}
EXPECTED = {  # model configuration of each configuration (resolved configs)
    # without RoPE, `vrope` (rotate_v) is a no-op: RotarySparseAttention.rope is None, so the
    # module is `standard` without RoPE, same parameters (5de0289b, 650b1d8d were trained so)
    "shape_knn_nopos": {
        "use_rope": "False",
        "attention_variant": ("standard", "vrope"),
    },
    "shape_knn_rope": {"use_rope": "True", "attention_variant": "standard"},
    "shape_knn": {"use_rope": "True", "attention_variant": "vrope"},
    "shape_knn_relpos": {
        "attention_variant": "rel_value"
    },  # RoPE on Q/K always (layer ignores use_rope)
}
COMMON = {"embedding_mode": "efd", "context": "knn", "k": "24"}


def resolved(client: MlflowClient, run_id: str) -> dict:
    dst = COMMIT_CHECK_DIR / run_id
    path = dst / "configs/config-resolved.yaml"
    if not path.exists():
        dst.mkdir(parents=True, exist_ok=True)
        client.download_artifacts(run_id, "configs/config-resolved.yaml", str(dst))
    return yaml.safe_load(path.read_text())


def model_config(cfg: dict) -> dict[str, str]:
    """Model settings of a resolved run config.

    Keys missing from old configs (before the context / attention-variant options, e.g. c5fe3bbb
    of 2026-08) take the code defaults: k-NN, RoPE, standard values.
    """
    net = cfg["model"]["net"]["config"]
    context = cfg["data"].get("context", "knn")
    return {
        "embedding_mode": str(cfg["data"]["dataset"]["embedding_mode"]),
        "net_embedding_mode": str(net.get("embedding_mode")),
        "context": str(context),
        "net_context": str(net.get("context", context)),
        "k": str(cfg["data"]["k"]),
        "block_size": str(cfg["data"]["block_size"]),
        "use_rope": str(net.get("use_rope", True)),
        "attention_variant": str(net.get("attention_variant", "standard")),
        "dim": str(net.get("dim")),
        "num_layers": str(net.get("num_layers")),
        "num_heads": str(net.get("num_heads")),
        "keys_defaulted": ",".join(
            k
            for k, present in (
                ("context", "context" in cfg["data"]),
                ("use_rope", "use_rope" in net),
                ("attention_variant", "attention_variant" in net),
            )
            if not present
        ),
    }


def find_eval_run(client: MlflowClient, train_prefix: str, eval_prefix: str) -> str:
    children = client.search_runs(
        [EXPERIMENT],
        filter_string=f"tags.mlflow.parentRunId LIKE '{train_prefix}%'",
        max_results=500,
    )
    hits = [r.info.run_id for r in children if r.info.run_id.startswith(eval_prefix)]
    assert len(hits) == 1, f"{eval_prefix}: {len(hits)} child runs of {train_prefix}*"
    return hits[0]


def download_checkpoint(client: MlflowClient, train_id: str) -> str:
    files = [f.path for f in client.list_artifacts(train_id, "checkpoints")]
    assert len(files) == 1, f"{train_id}: {len(files)} checkpoints"
    dst = COMMIT_CHECK_DIR / train_id
    if not list((dst / files[0]).rglob("*.ckpt")):
        client.download_artifacts(train_id, files[0], str(dst))
    (ckpt,) = list((dst / files[0]).rglob("*.ckpt"))
    return str(ckpt)


def main() -> None:
    client = MlflowClient()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    slides = pd.read_parquet(download_artifacts(METADATA_URI))
    slide_ids = sorted(slides["slide_id"])

    rows = []
    for name, pairs in RUNS.items():
        for seed, (tp, ep) in enumerate(pairs, start=1):
            eval_id = find_eval_run(client, tp, ep)
            ev = client.get_run(eval_id)
            train_id = ev.data.tags["mlflow.parentRunId"]
            assert train_id.startswith(tp)
            ev_cfg, tr_cfg = resolved(client, eval_id), resolved(client, train_id)
            # 4dda120e (2026-09): the run's config-resolved.yaml is that of a later metrics
            # step logged into the same run; the prediction config is not recorded
            eval_config_recorded = "model" in ev_cfg
            tr_model = model_config(tr_cfg)
            ev_model = model_config(ev_cfg) if eval_config_recorded else dict(tr_model)
            commit = next(
                (ev.data.tags[k] for k in COMMIT_TAGS if k in ev.data.tags), ""
            )
            check = checkpoint_check(client, eval_id, BLOCK_MASK_FIX_DATE)
            check.pop("logged")
            ckpt_run = re.search(r"/37/([0-9a-f]{32})/", str(ev_cfg.get("checkpoint")))
            _, info = load_run(eval_id, slide_ids, CACHE_DIR, "predictions")
            ckpt = download_checkpoint(client, train_id)
            expected = COMMON | EXPECTED[name]
            row = {
                "configuration": name,
                "seed": seed,
                "eval_run": eval_id,
                "training_run": train_id,
                "training_run_name": check["training_run_name"],
                "eval_commit": commit.split()[0][:8] if commit else None,
                "at_reference_commit": commit.startswith(REFERENCE_COMMIT),
                "eval_status": ev.info.status,
                "eval_date": check["eval_date"],
                "training_status": check["training_status"],
                "training_date": check["training_date"],
                # predictions come from a predict pass (metadata.predict); runs whose stored config
                # is the validate pass of the same MLflow run hold the test table under `test`
                "eval_config_recorded": eval_config_recorded,
                "eval_test_metadata_is_thesis": (
                    ev_cfg["data"]["metadata"].get(
                        "predict" if ev_cfg.get("mode") == "predict" else "test"
                    )
                    if eval_config_recorded
                    else ev_cfg.get("metadata_uri")  # the metrics step's test table
                )
                == METADATA_URI,
                "eval_mode": ev_cfg.get("mode"),
                **{f"train_{k}": v for k, v in tr_model.items()},
                **{f"eval_{k}": v for k, v in ev_model.items()},
                "train_keys_defaulted": tr_model.pop("keys_defaulted"),
                "eval_keys_defaulted": ev_model.pop("keys_defaulted"),
                "train_eval_model_equal": tr_model == ev_model,
                "train_eval_model_differ_in": ",".join(
                    k for k in tr_model if tr_model[k] != ev_model[k]
                ),
                "config_as_expected": all(
                    tr_model[k] in (v if isinstance(v, tuple) else (v,))
                    and ev_model[k] in (v if isinstance(v, tuple) else (v,))
                    for k, v in expected.items()
                )
                and tr_model["net_embedding_mode"] == "efd"
                and tr_model["net_context"] == "knn",
                "checkpoint_from_training_run": bool(ckpt_run)
                and ckpt_run[1] == train_id,
                "checkpoint_source": check["checkpoint_source"],
                "checkpoint_epoch": check["checkpoint_epoch"],
                "best_epoch": check["best_epoch"],
                "monitor": check["monitor"],
                "checkpoint_is_best": check["checkpoint_is_best"],
                "last_logged_epoch": check["last_logged_epoch"],
                "evaluated_before_block_mask_fix": check[
                    "evaluated_before_block_mask_fix"
                ],
                "trained_before_block_mask_fix": check["trained_before_block_mask_fix"],
                **training_schedule_check(client, train_id),
                **{f"predictions_{k}": v for k, v in info.items()},
                "local_checkpoint": Path(ckpt).relative_to(COMMIT_CHECK_DIR).as_posix(),
            }
            rows.append(row)
            print(
                f"{name} seed {seed}: eval {eval_id[:8]} <- training {train_id[:8]}, "
                f"commit {row['eval_commit']}, config ok {row['config_as_expected']}, "
                f"ckpt best {row['checkpoint_is_best']}, {info}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    text = df.to_csv(index=False)
    assert not CREDENTIAL_URL.search(text), "credential URL in the run table"
    assert not any(s in text for s in slide_ids), "slide id in the run table"
    (OUT_DIR / "preflight_runs.csv").write_text(text)
    print(
        json.dumps(
            {
                "runs": len(df),
                "config_as_expected": bool(df["config_as_expected"].all()),
                "train_eval_model_differ_in": df.loc[
                    ~df["train_eval_model_equal"],
                    ["eval_run", "train_eval_model_differ_in"],
                ]
                .assign(eval_run=lambda d: d.eval_run.str[:8])
                .values.tolist(),
                "schedule_check": df["schedule_check"].value_counts().to_dict(),
                "resumed_training_runs": df.loc[df["resumes"] > 0, "training_run"]
                .str[:8]
                .tolist(),
                "checkpoint_from_training_run": bool(
                    df["checkpoint_from_training_run"].all()
                ),
                "checkpoint_is_best": bool(df["checkpoint_is_best"].all()),
                "test_metadata_is_thesis": bool(
                    df["eval_test_metadata_is_thesis"].all()
                ),
                "complete_predictions": bool(
                    (df["predictions_test_slides_without_predictions"] == 0).all()
                    and (df["predictions_files_outside_test_metadata"] == 0).all()
                ),
                "commits": df.groupby("eval_commit")["eval_run"]
                .apply(lambda s: [r[:8] for r in s])
                .to_dict(),
            },
            indent=1,
        )
    )
    print(f"wrote {OUT_DIR / 'preflight_runs.csv'}")


if __name__ == "__main__":
    main()
