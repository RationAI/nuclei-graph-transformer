# ruff: noqa: RUF002  (Greek letters in text are intended)
r"""Single-concept removal from the concept bottleneck, within the epithelium and overall.

The k = 16 seed-1 SAE, for the encoder's codes κ (L1) and the NNLS refit κ̃ of
`xai.refit_magnitudes` separately; stored codes only.

Bottleneck: an unpenalised logistic regression of the nucleus label on the concept
activations (`xai.concept_subsets.fit_logistic`), fitted on all validation nuclei and
evaluated on the test nuclei, as the information curves of `xai.concept_subsets`. Per
concept j, one bottleneck on all 16 concepts and one on the 15 without j (refitted, not
mean-replaced). AUROC / AUPRC of both on the epithelial test nuclei (carcinoma vs benign
epithelium, the classes of `xai.epithelium_separation`) and on all test nuclei; the drop
= all 16 − without j.

95 % CIs: percentiles of a slide-level bootstrap (`bootstrap` resamples of the test slides
with replacement, the fits fixed). A resample weights every nucleus by its slide's
multiplicity; the metrics are the weighted AUROC / AUPRC (sklearn's definitions with
`sample_weight`, ties grouped), computed from one descending sort per score. The same
resamples are used for every score and variant, so the drops are paired.

Outputs in `out_dir_removal` (no slide ids): `removal.csv` (per variant and concept),
`bottleneck_all16.csv` (the all-16 metrics with CIs), `stable_epithelium_l1.tex` (the
drops for the stable concepts, κ, epithelium), `checks.json`.
"""

import json
import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from nuclei_graph.mlflow_utils import tag_parent_run
from nuclei_graph.xai.activations import load_activations
from nuclei_graph.xai.metrics import classification_metrics
from xai.concept_masks import code_commit
from xai.concept_subsets import concept_names, fit_logistic
from xai.epithelium_separation import BENIGN, CARCINOMA, bootstrap_counts, groups


SPLITS = ("validation", "test")
VARIANTS = ("l1", "refit")
METRICS = ("AUROC", "AUPRC")


class RankedScores:
    """Weighted AUROC / AUPRC of one score under many slide weightings."""

    def __init__(
        self, scores: NDArray, labels: NDArray, slide: NDArray, device: torch.device
    ) -> None:
        order = np.argsort(-scores, kind="stable")
        s = scores[order]
        ends = np.flatnonzero(np.r_[s[1:] != s[:-1], True])  # last row of each tie
        self.y = torch.from_numpy(labels[order].astype(np.float64)).to(device)
        self.slide = torch.from_numpy(slide[order].astype(np.int64)).to(device)
        self.ends = torch.from_numpy(ends).to(device)

    def __call__(self, weights: torch.Tensor) -> dict[str, torch.Tensor]:
        """`weights` (B, n_slides) → AUROC and AUPRC, each (B,)."""
        # few full-length (B, n) temporaries at a time: the GPU may be a small slice
        w = weights[:, self.slide]
        seen = torch.cumsum(w, 1)[:, self.ends]  # weighted rows at or above each cut
        w.mul_(self.y)
        tp = torch.cumsum(w, 1)[:, self.ends]
        del w
        fp = seen.sub_(tp)
        n_pos, n_neg = tp[:, -1:], fp[:, -1:]
        zero = torch.zeros_like(n_pos)
        ap = torch.diff(tp, dim=1, prepend=zero)
        ap.mul_(torch.where(tp + fp > 0, tp / (tp + fp), 0.0))
        ap = ap.sum(1) / n_pos[:, 0]
        tpr = torch.cat([zero, tp / n_pos], 1)
        del tp
        fpr = torch.cat([zero, fp / n_neg], 1)
        del fp
        auroc = (torch.diff(fpr, dim=1) * (tpr[:, 1:] + tpr[:, :-1])).sum(1) / 2
        return {"AUROC": auroc, "AUPRC": ap}


def evaluate(
    ranked: RankedScores, weights: torch.Tensor, batch: int
) -> dict[str, NDArray]:
    out: dict[str, list] = {m: [] for m in METRICS}
    for start in range(0, len(weights), batch):
        part = ranked(weights[start : start + batch])
        for m in METRICS:
            out[m].append(part[m].cpu().numpy())
    return {m: np.concatenate(v) for m, v in out.items()}


def compute(config: DictConfig) -> tuple[Path, dict]:
    torch.set_num_threads(config.num_threads)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device != "cpu" else "cpu"
    )
    out = Path(config.out_dir_removal)
    out.mkdir(parents=True, exist_ok=True)
    names, stable = concept_names(config.deck_key, list(config.stable_concepts))
    assert stable == sorted(config.stable_indices), (stable, config.stable_indices)
    k = 16
    order = sorted(stable, key=lambda j: names[j]) + [
        j for j in range(k) if j not in stable
    ]

    data = {s: load_activations(config.activations_dir, s) for s in SPLITS}
    labels = {s: data[s].labels for s in SPLITS}
    y_test = labels["test"]
    cls = np.load(config.epithelium_classes_test)
    assert len(cls) == len(y_test)
    group, label_checks = groups(y_test, cls)
    assert not any(label_checks.values()), label_checks
    epithelial = np.isin(group, (CARCINOMA, BENIGN))
    slide_ids = data["test"].index["slide_id"].astype(str).to_numpy()
    starts = np.flatnonzero(np.r_[True, slide_ids[1:] != slide_ids[:-1]])
    assert len(np.unique(slide_ids)) == len(starts), "test slides are not contiguous"
    slide = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(slide_ids)]))
    del slide_ids
    subsets = {"epithelium": epithelial, "all": np.ones(len(y_test), bool)}

    rng = np.random.default_rng(config.seed)
    weights = torch.from_numpy(bootstrap_counts(len(starts), config.bootstrap, rng)).to(
        device
    )
    ones = torch.ones(1, len(starts), dtype=torch.float64, device=device)

    stored = pd.read_csv(config.subsets_csv).query("subset == 'all'")
    stored = stored.set_index("variant")
    checks: dict = {
        "device": str(device),
        "n_test": len(y_test),
        "n_test_epithelial": int(epithelial.sum()),
        "carcinoma_prevalence_epithelial": float(y_test[epithelial].mean()),
        "n_test_slides": len(starts),
    }
    rows, all16_rows = [], []
    for variant in VARIANTS:
        codes = {s: np.load(config.codes[variant][s]) for s in SPLITS}
        for s in SPLITS:
            assert codes[s].shape == (len(data[s]), k), (variant, s)
        x_val = np.asarray(codes["validation"], np.float64)
        x_test = np.asarray(codes["test"], np.float64)
        del codes

        def fit_and_score(
            cols: list[int], x_val: NDArray = x_val, x_test: NDArray = x_test
        ) -> tuple[NDArray, dict]:
            start = time.time()
            coef, intercept, info = fit_logistic(
                x_val[:, cols], labels["validation"], device
            )
            info["seconds"] = time.time() - start
            return x_test[:, cols] @ coef + intercept, info

        def metrics_of(scores: NDArray) -> tuple[dict, dict]:
            point, reps = {}, {}
            for subset, mask in subsets.items():
                ranked = RankedScores(scores[mask], y_test[mask], slide[mask], device)
                p = ranked(ones)
                point[subset] = {m: float(p[m][0]) for m in METRICS}
                reps[subset] = evaluate(ranked, weights, config.batch)
                del ranked
            return point, reps

        scores_all, info_all = fit_and_score(list(range(k)))
        point_all, reps_all = metrics_of(scores_all)
        sk = {
            subset: classification_metrics(y_test[mask], scores_all[mask])
            for subset, mask in subsets.items()
        }
        checks[f"{variant}/all16_fit"] = info_all
        checks[f"{variant}/all16_gpu_minus_sklearn"] = {
            f"{subset}/{m}": point_all[subset][m] - sk[subset][m]
            for subset in subsets
            for m in METRICS
        }
        checks[f"{variant}/all16_all_vs_concept_subsets"] = {
            m: {
                "here": point_all["all"][m],
                "stored": float(stored.loc[variant, f"head_{m}"]),
            }
            for m in METRICS
        }
        for subset in subsets:
            row = {"variant": variant, "nuclei": subset}
            for m in METRICS:
                lo, hi = np.percentile(reps_all[subset][m], [2.5, 97.5])
                row |= {m: point_all[subset][m], f"{m}_ci_low": lo, f"{m}_ci_high": hi}
            all16_rows.append(row)
        print(variant, "all 16", point_all, flush=True)

        for rank, j in enumerate(order, 1):
            scores, info = fit_and_score([c for c in range(k) if c != j])
            point, reps = metrics_of(scores)
            row = {
                "variant": variant,
                "order": rank,
                "concept": names[j],
                "sae_index": j,
                "stable": j in stable,
            }
            for subset in subsets:
                for m in METRICS:
                    drop = reps_all[subset][m] - reps[subset][m]
                    lo, hi = np.percentile(drop, [2.5, 97.5])
                    row |= {
                        f"{m}_all16_{subset}": point_all[subset][m],
                        f"{m}_without_{subset}": point[subset][m],
                        f"d{m}_{subset}": point_all[subset][m] - point[subset][m],
                        f"d{m}_{subset}_ci_low": lo,
                        f"d{m}_{subset}_ci_high": hi,
                    }
            row |= {
                "fit_iterations": info["iterations"],
                "fit_grad_max": info["grad_max"],
            }
            rows.append(row)
            print(
                variant,
                names[j],
                f"{info['seconds']:.1f}s",
                {
                    c: round(row[c], 4)
                    for c in (
                        "dAUPRC_epithelium",
                        "dAUPRC_epithelium_ci_low",
                        "dAUPRC_epithelium_ci_high",
                        "dAUPRC_all",
                    )
                },
                flush=True,
            )

    removal = pd.DataFrame(rows)
    removal.to_csv(out / "removal.csv", index=False)
    pd.DataFrame(all16_rows).to_csv(out / "bottleneck_all16.csv", index=False)
    (out / "stable_epithelium_l1.tex").write_text(latex(removal))
    (out / "checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    return out, checks


def fmt(value: float) -> str:
    """Three decimals in math mode (a proper minus); no "-0.000"."""
    return f"${round(value, 3) + 0.0:.3f}$"


def latex(removal: pd.DataFrame) -> str:
    """Stable concepts, κ, epithelial drop of AUPRC with its 95 % CI (three decimals)."""
    part = removal.query("variant == 'l1' and stable").sort_values("concept")
    lines = [
        "% concept & ΔAUPRC epithelium (κ, bottleneck, all 16 minus without j) & 95 % CI",
    ]
    for r in part.itertuples(index=False):
        lines.append(
            f"{r.concept} & {fmt(r.dAUPRC_epithelium)} & "
            f"[{fmt(r.dAUPRC_epithelium_ci_low)}, {fmt(r.dAUPRC_epithelium_ci_high)}] \\\\"
        )
    return "\n".join(lines) + "\n"


def metric_name(concept: str) -> str:
    return f"u{concept[1:]}" if concept.startswith("#") else concept


@with_cli_args(["+xai=bottleneck_removal"])
@hydra.main(config_path="../configs", config_name="xai", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)
    client = logger.experiment
    commit, dirty = code_commit("xai", "nuclei_graph", "configs")
    client.set_tag(
        logger.run_id, "xai.code_commit", f"{commit}{' (dirty)' if dirty else ''}"
    )
    for key, value in config.links.items():
        if value is not None:
            client.set_tag(logger.run_id, f"xai.{key}", value)
    out, _ = compute(config)
    for row in pd.read_csv(out / "bottleneck_all16.csv").to_dict("records"):
        logger.log_metrics(
            {
                f"test/{row['variant']}/{row['nuclei']}/all16/{m}": float(row[m])
                for m in METRICS
            }
        )
    for row in pd.read_csv(out / "removal.csv").to_dict("records"):
        prefix = f"test/{row['variant']}/{metric_name(row['concept'])}"
        logger.log_metrics(
            {
                f"{prefix}/d{m}_{subset}": float(row[f"d{m}_{subset}"])
                for m in METRICS
                for subset in ("epithelium", "all")
            }
        )
    for name in (
        "removal.csv",
        "bottleneck_all16.csv",
        "stable_epithelium_l1.tex",
        "checks.json",
    ):
        logger.log_artifact(str(out / name))
    for source in (__file__, "configs/xai/bottleneck_removal.yaml"):
        logger.log_artifact(str(Path(__file__).parents[1] / source), "code")


if __name__ == "__main__":
    main()
