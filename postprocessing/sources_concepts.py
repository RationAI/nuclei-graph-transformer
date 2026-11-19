# ruff: noqa: RUF002  (mathematical symbols in text are intended)
r"""fig:res-sources-concepts: which information source fits which nuclei, by dominating SAE concept.

PANDA Radboud test set, all 14.7 M test nuclei. Stored outputs only: the test predictions of the
five configurations of the "sources" family of tab:res-sources (`postprocessing.results_sources`,
run a925ae67: shape and texture, self-only and k-NN with rotated values; position only, k-NN with
rotated values; k = 24, three seeds each) and the stored SAE contributions of the thesis model
c8f26853. No training, no inference.

Grouping: exactly that of fig:res-context-concepts (`postprocessing.context_concepts`, run
875566dc). The same code on the same stored φ_ij = κ_ij wᵀu_j assigns each nucleus to the stable
concept (A-H) with the largest |φ_ij|, or "other". 875566dc's per-group nucleus and carcinoma
counts must be reproduced exactly. The assignment is saved as `dominant_group.npy` (int8 index
into `checks.json["groups"]`, rows in the predictions' (slide, nucleus key) order).

Per group and configuration: the mean log-loss of the seed-averaged probability (clip 1e-7).
Differences of group means, first − second (negative = the first has the lower loss): shape −
texture self-only and with context, shape − position only, texture − position only. The context
gains of shape and texture (k-NN − self-only) go in the CSV only. Percentile CIs come from the
stratified slide bootstrap of the thesis (seed 20261108, B = 2,000). p = 2 x the smaller tail
fraction, unadjusted (the figure is descriptive, no Holm).

Outputs in `out_dir` (no slide ids): `groups.csv`, `replicates.npz`, `checks.json`,
`dominant_group.npy`, `figures/res_sources_concepts.{pdf,png}`.
"""

import json
import subprocess
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from nuclei_graph.mlflow_utils import tag_parent_run
from postprocessing.context_concepts import group_table, load_grouped
from postprocessing.results_context import REPO
from visualization.sources_concepts_figure import (
    FIGURE_STEM,
    save_sources_concepts_figure,
)


SOURCES = (
    "postprocessing/sources_concepts.py",
    "postprocessing/context_concepts.py",
    "postprocessing/bootstrap_sources.py",
    "visualization/sources_concepts_figure.py",
    "configs/postprocessing/sources_concepts.yaml",
    "configs/postprocessing/context_concepts.yaml",
    "configs/postprocessing/results_context.yaml",
    "configs/postprocessing/results_sources.yaml",
)


def code_commit() -> str:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    dirty = git("status", "--porcelain", "--untracked-files=no", "--", *SOURCES)
    return f"{git('rev-parse', 'HEAD')}{' (dirty)' if dirty else ''}"


def check_runs(config: DictConfig) -> None:
    """The evaluation runs are those of tab:res-sources."""
    reference = OmegaConf.load(REPO / config.sources_config).configurations
    for name in config.configurations_used:
        runs = list(config.configurations[name].eval_runs)
        assert runs == list(reference[name].eval_runs), f"{name}: runs differ"
        assert len(runs) == 3, f"{name}: {len(runs)} seeds"


def check_reference_groups(config: DictConfig, table: pd.DataFrame) -> dict:
    """Per-group nucleus and carcinoma counts identical to the context-concepts run."""
    reference = pd.read_csv(config.reference_groups)
    cols = ["group", "n_nuclei", "n_carcinoma"]
    assert table["group"].tolist() == reference["group"].tolist(), "groups differ"
    assert table[cols].equals(reference[cols]), "group counts differ from the reference"
    return {
        "reference_run": config.reference_run,
        "reference_groups": str(config.reference_groups),
        "group_counts_identical": True,
    }


def compute(config: DictConfig) -> tuple[Path, dict]:
    out = Path(config.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    check_runs(config)
    data = load_grouped(config)
    table, diff_reps, checks = group_table(config, data)
    checks["grouping"] = check_reference_groups(config, table)
    assert checks["resamples"]["replicate_group_cells_without_nuclei"] == 0, (
        "a group is empty in some resample"
    )
    checks["configurations"] = {
        n: list(config.configurations[n].eval_runs) for n in config.configurations_used
    }

    table.to_csv(out / "groups.csv", index=False)
    np.savez_compressed(out / "replicates.npz", **diff_reps)  # type: ignore[arg-type]
    np.save(out / "dominant_group.npy", data["group"])
    (out / "checks.json").write_text(json.dumps(checks, indent=1, default=str))

    figure_dirs = [out / "figures"]
    if config.get("repo_figures_dir"):
        figure_dirs.append(Path(config.repo_figures_dir))
    save_sources_concepts_figure(out, figure_dirs)
    return out, checks


@with_cli_args(["+postprocessing=sources_concepts"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)
    client = logger.experiment
    client.set_tag(logger.run_id, "code_commit", code_commit())
    for key, value in config.links.items():
        client.set_tag(logger.run_id, key, value)
    out, _ = compute(config)
    table = pd.read_csv(out / "groups.csv")
    metrics = {}
    for r in table.to_dict("records"):
        g = r["group"]
        metrics[f"{g}/n_nuclei"] = float(r["n_nuclei"])
        metrics[f"{g}/carcinoma_share"] = float(r["carcinoma_share"])
        for n in config.configurations_used:
            metrics[f"{g}/log_loss_{n}"] = r[f"log_loss_{n}"]
        for d in config.differences:
            for suffix in ("", "_ci_lo", "_ci_hi", "_p"):
                metrics[f"{g}/{d.name}{suffix}"] = r[f"{d.name}{suffix}"]
    logger.log_metrics(metrics)
    for name in ("groups.csv", "checks.json", "replicates.npz"):
        logger.log_artifact(str(out / name))
    for ext in ("pdf", "png"):
        logger.log_artifact(str(out / "figures" / f"{FIGURE_STEM}.{ext}"), "figures")
    for source in SOURCES:
        logger.log_artifact(str(REPO / source), "code")
    (out / "config-resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    logger.log_artifact(str(out / "config-resolved.yaml"), "code")


if __name__ == "__main__":
    main()
