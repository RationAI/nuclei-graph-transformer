"""Sets up the XAI folder of another training seed of the analysed model.

For `xai.cross_model_concepts`: the activations of the new model must cover exactly the
nuclei of the analysed model (same validation and test slides, same row order), so its
`slides.parquet` and `activations/<split>/index.parquet` are copied from the analysed
model's folder instead of being recomputed. `slides_predict.parquet` is the same table
with the nuclei paths remapped (`path_map`) for the machine that runs the extraction.

The new run is checked against the analysed run: every architecture, data and split
parameter in `CHECKED_PARAMS` must be equal (the training seed may differ). The run must
have exactly one checkpoint (`save_top_k: 1`, the lowest validation loss); its URI is
written to `source.json` for the activation extraction.

    python -m xai.cross_model_prepare <run_id> <reference_dir> <out_dir> [<from>=<to> ...]
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from mlflow.tracking import MlflowClient


CHECKED_PARAMS = [
    "k",
    "context",
    "embedding_mode",
    "attention_variant",
    "use_rope",
    "efd_order",
    "dim",
    "hidden_dim",
    "num_heads",
    "num_layers",
    "drop_path_rate",
    "split_size",
    "batch_size",
    "lr",
    "warmup_epochs",
    "crop_size_min",
    "crop_size_max",
    "train_supervision_mode",
    "validation_supervision_mode",
]


def checkpoints(
    client: MlflowClient, run_id: str, path: str = "checkpoints"
) -> list[str]:
    out = []
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            out += checkpoints(client, run_id, item.path)
        elif item.path.endswith(".ckpt"):
            out.append(item.path)
    return out


def main(
    run_id: str, reference_dir: Path, out_dir: Path, path_map: dict[str, str]
) -> None:
    client = MlflowClient()
    reference_id = reference_dir.name
    run, reference = client.get_run(run_id), client.get_run(reference_id)
    params, reference_params = run.data.params, reference.data.params
    differences = {
        key: (reference_params.get(key), params.get(key))
        for key in CHECKED_PARAMS
        if reference_params.get(key) != params.get(key)
    }
    assert not differences, f"{run_id} differs from {reference_id}: {differences}"
    assert run.info.experiment_id == reference.info.experiment_id

    found = checkpoints(client, run_id)
    assert len(found) == 1, f"{run_id}: expected one checkpoint, found {found}"
    source = {
        "run_id": run_id,
        "run_name": run.info.run_name,
        "status": run.info.status,
        "reference_run_id": reference_id,
        "checkpoint": f"mlflow-artifacts:/{run.info.experiment_id}/{run_id}/artifacts/{found[0]}",
        "seed": params.get("seed"),
        "reference_seed": reference_params.get("seed"),
        "git_commit": run.data.tags.get("mlflow.source.git.commit"),
        "checked_params": {key: params.get(key) for key in CHECKED_PARAMS},
    }
    print(json.dumps(source, indent=2))

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(reference_dir / "slides.parquet", out_dir / "slides.parquet")
    slides = pd.read_parquet(out_dir / "slides.parquet")
    for old, new in path_map.items():
        slides["slide_nuclei_path"] = slides["slide_nuclei_path"].str.replace(
            old, new, regex=False
        )
    assert all(Path(p).exists() for p in slides["slide_nuclei_path"]), (
        "Nuclei paths not found"
    )
    slides.to_parquet(out_dir / "slides_predict.parquet", index=False)
    for split in ("validation", "test"):
        (out_dir / "activations" / split).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            reference_dir / "activations" / split / "index.parquet",
            out_dir / "activations" / split / "index.parquet",
        )
    (out_dir / "source.json").write_text(json.dumps(source, indent=2) + "\n")


if __name__ == "__main__":
    main(
        sys.argv[1],
        Path(sys.argv[2]),
        Path(sys.argv[3]),
        dict(arg.split("=", 1) for arg in sys.argv[4:]),
    )
