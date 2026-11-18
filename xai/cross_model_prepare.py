"""Sets up the XAI folder of another training seed of the analysed model.

For `xai.cross_model_concepts`: the activations of the new model must cover exactly the
nuclei of the analysed model (same validation and test slides, same row order), so its
`slides.parquet` and `activations/<split>/index.parquet` are copied from the analysed
model's folder instead of being recomputed. `slides_predict.parquet` is the same table
with the nuclei paths remapped (`path_map`) for the machine that runs the extraction.

The new run is checked against the analysed run: the `model`, `data` and `trainer`
sections of their resolved training configs must be equal (the training seed may differ;
the logged params are not used, older runs lack some of them). The run must
have exactly one checkpoint (`save_top_k: 1`, the lowest validation loss); its URI is
written to `source.json` for the activation extraction.

    python -m xai.cross_model_prepare <run_id> <reference_dir> <out_dir> [<from>=<to> ...]
"""

import json
import shutil
import sys
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf


CHECKED_SECTIONS = ["model", "data", "trainer"]  # of the resolved training config


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
    configs = {
        key: OmegaConf.to_container(
            OmegaConf.load(
                mlflow.artifacts.download_artifacts(
                    run_id=key, artifact_path="configs/config-resolved.yaml"
                )
            )
        )
        for key in (run_id, reference_id)
    }
    differences = {
        section: (configs[reference_id].get(section), configs[run_id].get(section))
        for section in CHECKED_SECTIONS
        if configs[reference_id].get(section) != configs[run_id].get(section)
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
        "seed": configs[run_id].get("seed"),
        "reference_seed": configs[reference_id].get("seed"),
        "init_checkpoint": configs[run_id].get("checkpoint"),
        "git_commit": run.data.tags.get("mlflow.source.git.commit"),
        "checked_sections": CHECKED_SECTIONS,
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
