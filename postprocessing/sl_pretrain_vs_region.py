r"""Slide-label (sl_pretrain) vs region-supervised model: nucleus-level, PANDA Radboud test.

Both on the same test nuclei. Inference outputs only: the per-nucleus test predictions of the
evaluation (prediction) runs, `<run>/artifacts/predictions/<slide_id>.parquet` (`id`,
`nuclei_prediction`); the slide-label model's come from
`configs/experiment/modeling/prediction/graph_level/sl_pretrain/radboud_test.yaml`.

Everything follows the thesis nucleus-level evaluation (`postprocessing.nucleus_results`,
`postprocessing.bootstrap_sources`) and reuses its helpers:

1. Labels: dense annotation labels left-joined on (slide, nucleus), missing -> 0, every nucleus
   of an `is_carcinoma = False` slide -> 0.
2. Nuclei: the intersection of the runs' nuclei (normally all of them; the drop is reported).
3. Metrics: exact AUPRC / AUROC over all test nuclei pooled, per run; a configuration (several
   seeds) is the mean over its runs.
4. Paired bootstrap over the canonical stratified slide resamples of Radboud test (the Q2-Q4 file,
   B = 2,000): per resample, each run's metric from per-slide logit histograms (grid refined
   until every run's full-test-set metrics match the exact values within the tolerance), seeds
   averaged per configuration, then the difference slide-label - region. Percentile CIs and a
   two-sided bootstrap p.

Outputs (`out_dir`, logged to one MLflow run): `configurations.csv`, `comparison.csv`,
`runs.csv`, `checks.json`. No slide ids are written.
"""

import json
import re
import subprocess
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig, OmegaConf
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from postprocessing.bootstrap_sources import (
    METRICS,
    bin_logits,
    bootstrap_p,
    exact_metrics,
    hist_metrics,
    label_predictions,
    load_labels,
    load_run,
    replicate_metrics,
    slide_histograms,
    to_logits,
)


CREDENTIAL_URL = re.compile(r"https?://[^/\s\"']*@")


def run_cache(run_id: str, caches: list[Path]) -> Path:
    """The first cache holding the run's aligned predictions, else the last (writable) one."""
    for cache in caches:
        if (cache / f"{run_id}.parquet").exists() and (
            cache / f"{run_id}.json"
        ).exists():
            return cache
    return caches[-1]


def compute(config: DictConfig) -> Path:
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    caches = [Path(c) for c in config.cache_dirs]

    slides = pd.read_parquet(download_artifacts(config.metadata_uri))[
        ["slide_id", "is_carcinoma"]
    ].sort_values("slide_id", ignore_index=True)
    slide_ids = slides["slide_id"].tolist()
    carcinoma = slides["is_carcinoma"].to_numpy(bool)
    labels = load_labels(slides, Path(config.supervision_dir), config.label_column)

    names = list(config.configurations)
    assert names == ["slide_label", "region"], names
    runs = [(n, r) for n in names for r in config.configurations[n].eval_runs]
    assert len({r for _, r in runs}) == len(runs), "an evaluation run is listed twice"

    frames, rows = [], []
    for name, run_id in runs:
        df, info = load_run(
            run_id, slide_ids, run_cache(run_id, caches), config.predictions_artifact
        )
        frames.append(df)
        rows.append({"configuration": name, "eval_run": run_id, **info})

    common = frames[0][["slide", "key"]]
    for f in frames[1:]:
        common = common.merge(f[["slide", "key"]], on=["slide", "key"], how="inner")
    common = common.sort_values(["slide", "key"], ignore_index=True)
    probs = np.stack(
        [
            common.merge(f, on=["slide", "key"], how="left")["p"].to_numpy()
            for f in frames
        ]
    )
    for row, f in zip(rows, frames, strict=True):
        row["nuclei_dropped_by_intersection"] = len(f) - len(common)
    y, missing = label_predictions(common, labels, carcinoma)
    slide = common["slide"].to_numpy()
    n_slides = len(slides)
    assert np.bincount(slide, minlength=n_slides).min() > 0, (
        "a test slide has no nuclei"
    )
    positive = np.bincount(slide, weights=(y == 1), minlength=n_slides) > 0

    exact = [exact_metrics(y, p) for p in probs]
    for row, e in zip(rows, exact, strict=True):
        row |= {f"{m}_exact": e[m] for m in METRICS}

    r = np.load(config.resamples)
    assert int(r["seed"]) == int(config.resamples_seed)
    assert list(r["slide_ids"]) == slide_ids, "resamples are for other slides"
    assert np.array_equal(r["positive"], positive), "resample strata differ"
    weights = r["multiplicities"].astype(np.float64)

    logits = [to_logits(p) for p in probs]
    finite = np.concatenate([lg[np.isfinite(lg)] for lg in logits])
    lo, hi = float(finite.min()), float(finite.max())
    del finite
    n_bins = int(config.histogram.n_bins)
    while True:
        hists, max_err = [], 0.0
        for lg, e in zip(logits, exact, strict=True):
            pos, neg = slide_histograms(
                slide, bin_logits(lg, lo, hi, n_bins), y, n_slides, n_bins
            )
            full = hist_metrics(pos.sum(0), neg.sum(0))
            max_err = max(max_err, *(abs(float(full[m] - e[m])) for m in METRICS))
            hists.append((pos, neg))
        print(f"histogram {n_bins} bins: max |error| {max_err:.2e}")
        if max_err <= config.histogram.tolerance or n_bins >= config.histogram.max_bins:
            break
        n_bins *= 2
    assert max_err <= config.histogram.tolerance, "histogram grid too coarse"

    rep = np.stack(
        [
            np.stack([replicate_metrics(pos, neg, weights)[m] for m in METRICS])
            for pos, neg in hists
        ]
    )  # (runs, metrics, resamples)
    del hists
    idx = {n: [i for i, (c, _) in enumerate(runs) if c == n] for n in names}
    q = (1 - config.ci_level) / 2 * 100
    conf_rows, comp_rows = [], []
    for n in names:
        row = {
            "configuration": n,
            "label": config.configurations[n].label,
            "runs": len(idx[n]),
        }
        for k, m in enumerate(METRICS):
            boot = rep[idx[n], k].mean(0)
            row |= {
                m: float(np.mean([exact[i][m] for i in idx[n]])),
                f"{m}_ci_lo": float(np.percentile(boot, q)),
                f"{m}_ci_hi": float(np.percentile(boot, 100 - q)),
            }
        conf_rows.append(row)
    for k, m in enumerate(METRICS):
        diff = rep[idx["slide_label"], k].mean(0) - rep[idx["region"], k].mean(0)
        point = conf_rows[0][m] - conf_rows[1][m]
        comp_rows.append(
            {
                "comparison": "slide_label - region",
                "metric": m,
                "difference": point,
                "ci_lo": float(np.percentile(diff, q)),
                "ci_hi": float(np.percentile(diff, 100 - q)),
                "p": bootstrap_p(diff),
            }
        )

    pd.DataFrame(conf_rows).to_csv(out / "configurations.csv", index=False)
    pd.DataFrame(comp_rows).to_csv(out / "comparison.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
    checks = {
        "test_slides": n_slides,
        "nuclei": len(common),
        "positive_nuclei": int((y == 1).sum()),
        "carcinoma_slide_nuclei_without_supervision_row": missing,
        "histogram": {
            "n_bins": n_bins,
            "logit_range": [lo, hi],
            "max_abs_error": max_err,
        },
        "resamples": {"file": str(config.resamples), "n_resamples": len(weights)},
    }
    (out / "checks.json").write_text(json.dumps(checks, indent=1))
    for path in out.iterdir():  # no credentials, no slide ids
        if path.suffix in {".csv", ".json"}:
            text = path.read_text()
            assert not CREDENTIAL_URL.search(text), path.name
            assert not any(s in text for s in slide_ids), path.name
    return out


@with_cli_args(["+postprocessing=sl_pretrain_vs_region"])
@hydra.main(config_path="../configs", config_name="postprocessing", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    repo = Path(__file__).parents[1]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    logger.experiment.set_tag(logger.run_id, "code_commit", commit)
    out = compute(config)
    comp = pd.read_csv(out / "comparison.csv")
    conf = pd.read_csv(out / "configurations.csv")
    logger.log_metrics(
        {
            f"test/{r.configuration}/{m}": float(getattr(r, m))
            for r in conf.itertuples()
            for m in METRICS
        }
        | {
            f"test/difference/{r.metric}": float(r.difference)
            for r in comp.itertuples()
        }
    )
    (out / "config-resolved.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
    for name in (
        "configurations.csv",
        "comparison.csv",
        "runs.csv",
        "checks.json",
        "config-resolved.yaml",
    ):
        logger.log_artifact(str(out / name))


if __name__ == "__main__":
    main()
