# ruff: noqa: RUF002  (mathematical symbols in text are intended)
r"""tab:res-context-probes: how well each context configuration's h predicts the concepts.

Targets: the activations κ_j of the stable concepts A-H of the analysed SAE (k = 16, L1, seed 1;
SAE training run 4cf1db8f) of the thesis model c8f26853, stored per nucleus
(`concepts/k16_seed1/morphometry/codes_validation.npy`, `codes_test.npy`; checked against
encoding the analysed model's stored h of the sampled rows).

Inputs: the final representations h (output of `final_norm`, float16) of the self-only, k-NN
(no position encoding) and dense models, three seeds each, extracted with each evaluation run's
code and mask (scripts/reporting/context_probes_extract.py, whose `check.csv` compares the
re-predicted test probabilities with the stored ones), on
- a fixed uniform random sample of 2,000,000 validation nuclei of the analysed model (seed
  20261119; no slide above `max_slide_share` = 1 % of it, asserted), where the probes are fitted;
- every test nucleus (14,730,504), where they are evaluated.
Reference: the analysed model's own stored h on the same nuclei.

Probe: per model and concept, ordinary least squares with intercept from h (256 dims) to κ_j,
fitted on the validation sample; R² on all test nuclei, 1 − Σ(κ − κ̂)² / Σ(κ − mean κ)², the mean
taken over the test nuclei. A configuration's value is the mean of its seeds' R².

Outputs in `out_dir` (no slide ids): `probes.csv` (model x concept: R² test and validation),
`configurations.csv` (seed means, SDs, per seed), `tab_res_context_probes.tex`, `checks.json`.
"""

import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from nuclei_graph.mlflow_utils import tag_parent_run
from nuclei_graph.xai.sae import encode, load_autoencoder
from postprocessing.bootstrap_sources import nucleus_keys
from xai.concept_morphometry import concept_letters


REPO = Path(__file__).parents[1]
PARTS = {  # split -> (rows file, output tag) of scripts/reporting/context_probes_extract.py
    "validation": ("sample_validation.parquet", "validation"),
    "test": ("test_rows.parquet", "test_all"),
}
SOURCES = (
    "postprocessing/context_probes.py",
    "postprocessing/nucleus_inference_at_commit.py",
    "scripts/reporting/context_probes_extract.py",
    "configs/postprocessing/context_probes.yaml",
)


def code_commit() -> str:
    import subprocess

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    dirty = git("status", "--porcelain", "--untracked-files=no", "--", *SOURCES)
    return f"{git('rev-parse', 'HEAD')}{' (dirty)' if dirty else ''}"


def ols_fit(x: NDArray, y: NDArray) -> NDArray:
    """Least-squares coefficients (intercept first) of y (n, m) on x (n, d), in float64."""
    design = np.hstack([np.ones((len(x), 1)), x.astype(np.float64)])
    coef, *_ = np.linalg.lstsq(design, y.astype(np.float64), rcond=None)
    return coef


def r2(h: NDArray, kappa: NDArray, coef: NDArray, chunk: int = 2_000_000) -> NDArray:
    """R² of the probe on all rows, 1 − Σ(κ − κ̂)² / Σ(κ − mean κ)², in chunks (float64)."""
    mean = kappa.mean(0, dtype=np.float64)
    ss_res = np.zeros(kappa.shape[1])
    ss_tot = np.zeros(kappa.shape[1])
    for start in range(0, len(kappa), chunk):
        y = kappa[start : start + chunk].astype(np.float64)
        pred = coef[0] + h[start : start + chunk].astype(np.float64) @ coef[1:]
        ss_res += ((y - pred) ** 2).sum(0)
        ss_tot += ((y - mean) ** 2).sum(0)
    return 1 - ss_res / ss_tot


def compute(config: DictConfig) -> tuple[Path, dict]:
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(config.work_dir)
    x = config.xai

    letters = dict(concept_letters(x.deck_key))
    stable = sorted(j for j, letter in letters.items() if letter in x.stable_concepts)
    assert stable == sorted(x.stable_indices), (stable, x.stable_indices)
    order = sorted(stable, key=lambda j: letters[j])  # A..H
    names = [letters[j] for j in order]

    # nuclei, targets, the analysed model's h ----------------------------------------
    # validation: the 2M sample (fit); test: every nucleus (evaluation). Rows of both are in
    # the (slide, key) order of the extraction outputs; xai_row points into the XAI arrays.
    model_sae, meta = load_autoencoder(x.sae_model)
    assert (meta["k"], meta["seed"]) == (16, 1), meta
    kappa, h_ref, checks = {}, {}, {}
    for split, (rows_file, _) in PARTS.items():
        s = pd.read_parquet(work / rows_file)
        rows = s["xai_row"].to_numpy()
        index = pd.read_parquet(x.index[split], columns=["nucleus_id"])
        assert np.array_equal(
            nucleus_keys(index["nucleus_id"].iloc[rows]), s["key"].to_numpy()
        ), f"{split}: rows do not match the XAI index"
        del index
        codes = np.load(x.codes[split], mmap_mode="r")
        h_all = np.load(x.h[split], mmap_mode="r")
        assert len(codes) == len(h_all)
        kappa[split] = np.asarray(codes[rows])[:, order]
        h_ref[split] = np.asarray(h_all[rows])
        checks[f"{split}/nuclei"] = len(s)
        checks[f"{split}/nuclei_in_split"] = len(h_all)
        checks[f"{split}/slides"] = int(s["slide"].nunique())
        if split == "validation":
            share = float(s["slide"].value_counts().max() / len(s))
            assert share <= config.max_slide_share, share
            checks["validation/max_slide_share"] = share
        else:
            assert len(s) == len(h_all), "test: not every nucleus"
        # stored κ vs encoding the stored h, on (up to) 2M rows of the split
        check_rows = np.linspace(0, len(s) - 1, min(len(s), 2_000_000)).astype(int)
        encoded = (
            encode(model_sae, torch.as_tensor(h_ref[split][check_rows]).float())
            .cpu()
            .numpy()
        )
        checks[f"{split}/stored_codes_vs_encoded_max_abs_diff"] = float(
            np.abs(encoded[:, order] - kappa[split][check_rows]).max()
        )
        checks[f"{split}/active_fraction"] = dict(
            zip(names, (kappa[split] > 0).mean(0).round(6).tolist(), strict=True)
        )
    assert all(
        v <= x.code_tolerance for k, v in checks.items() if k.endswith("max_abs_diff")
    ), checks

    # probes -------------------------------------------------------------------------
    models = pd.read_csv(work / "models.csv")
    extraction = pd.read_csv(work / "check.csv").set_index("eval_run")
    assert len(extraction) == len(models), "extraction check incomplete"
    entries = [("reference", 1, config.reference.model_run, None)] + [
        (r.configuration, int(r.seed), r.eval_run, r.eval_run)
        for r in models.itertuples()
    ]
    rows = []
    for name, seed, run_id, eval_run in entries:
        h = (
            h_ref
            if eval_run is None
            else {
                split: np.load(work / "outputs" / f"{eval_run}_{tag}_h.npy")
                for split, (_, tag) in PARTS.items()
            }
        )
        for split in PARTS:
            assert len(h[split]) == len(kappa[split]), (run_id, split)
        coef = ols_fit(h["validation"], kappa["validation"])
        r2_test = r2(h["test"], kappa["test"], coef)
        r2_val = r2(h["validation"], kappa["validation"], coef)
        del h
        for j, concept in enumerate(names):
            rows.append(
                {
                    "configuration": name,
                    "seed": seed,
                    "run": run_id,
                    "concept": concept,
                    "r2_test": float(r2_test[j]),
                    "r2_validation": float(r2_val[j]),
                }
            )
        print(name, seed, np.round(r2_test, 3), flush=True)
    probes = pd.DataFrame(rows)
    probes.to_csv(out / "probes.csv", index=False)

    conf = (
        probes.groupby(["configuration", "concept"])["r2_test"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    per_seed = probes.pivot_table(
        index=["configuration", "concept"], columns="seed", values="r2_test"
    ).add_prefix("seed_")
    conf = conf.merge(per_seed.reset_index(), on=["configuration", "concept"])
    conf.to_csv(out / "configurations.csv", index=False)

    means = conf.pivot(index="configuration", columns="concept", values="mean")[names]
    lines = [
        "% Generated by postprocessing/context_probes.py (PANDA Radboud; probes of the thesis model's",
        "% SAE concepts A-H, k = 16 L1 seed 1). Test R^2 of OLS probes (with intercept) from the final",
        (
            f"% representation h, fitted on {checks['validation/nuclei']:,} random validation nuclei"
            f" (<= 1 % per slide), evaluated on all {checks['test/nuclei']:,} test nuclei; mean of three seeds"
            " (reference: the analysed model itself)."
        ),
    ]
    for key, label in config.table_rows.items():
        cells = " & ".join(f"{means.loc[key, c]:.3f}" for c in names)
        lines.append(f"    {label} & {cells} \\\\")
    (out / "tab_res_context_probes.tex").write_text("\n".join(lines) + "\n")

    checks["concepts"] = names
    checks["extraction"] = {
        "max_abs_p_diff": float(extraction["max_abs_p_diff"].max()),
        "mean_abs_p_diff_max": float(extraction["mean_abs_p_diff"].max()),
        "class_head_max_abs_diff": float(
            extraction[[f"class_head_max_abs_diff_{t}" for _, t in PARTS.values()]]
            .to_numpy()
            .max()
        ),
    }
    (out / "checks.json").write_text(json.dumps(checks, indent=1))
    return out, checks


@with_cli_args(["+postprocessing=context_probes"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)
    client = logger.experiment
    client.set_tag(logger.run_id, "code_commit", code_commit())
    for key, value in config.links.items():
        client.set_tag(logger.run_id, key, value)
    out, _ = compute(config)
    conf = pd.read_csv(out / "configurations.csv")
    logger.log_metrics(
        {f"r2_test/{r.configuration}/{r.concept}": r.mean for r in conf.itertuples()}
    )
    for name in (
        "probes.csv",
        "configurations.csv",
        "tab_res_context_probes.tex",
        "checks.json",
    ):
        logger.log_artifact(str(out / name))
    work = Path(config.work_dir)
    for name in ("models.csv", "check.csv", "pilot.csv", "sample.json"):
        if (work / name).exists():
            logger.log_artifact(str(work / name), "extraction")
    for source in SOURCES:
        logger.log_artifact(str(REPO / source), "code")
    (out / "config-resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    logger.log_artifact(str(out / "config-resolved.yaml"), "code")


if __name__ == "__main__":
    main()
