r"""Concept bottleneck of the k = 16 seed-1 SAE: linear probes on nested stable subsets.

A bottleneck model sees only the activations of a concept subset S; the other concepts
are dropped (not mean-replaced, as in `xai.concept_subsets`). S starts from the stable
eight (A–H) and shrinks by dropping the least important remaining stable concept, by the
L1 validation importance share mean |φ_j| / Σ_k mean |φ_k| (`ranking_variant`), down to
one concept. All 16 are a reference row. Done separately for the encoder's codes κ (L1)
and the NNLS refit κ̃ of `xai.refit_magnitudes`, with the same subsets. No retraining of
the SAE or the model.

Two linear probes per variant and subset, fitted on all validation nuclei, evaluated on
all test nuclei:

- label probe (the prediction task): unpenalised logistic regression of the nucleus label
  on κ_S (`xai.concept_subsets.fit_logistic`). AUROC, AUPRC and log loss on test.
- logit probe (logit reconstruction): least squares f ≈ β₀ + Σ_{j∈S} β_j κ_j of the
  model's logit f (OLS, with intercept). On test: MCE_subset = RMSE(f, f̂),
  MCE_subset / std(f), R², Spearman(f, f̂), class flips sign(f) ≠ sign(f̂) on all nuclei
  and among |f| < `flip_margin`, AUROC / AUPRC of f̂ against the nucleus labels;
  MCE_subset on validation (in sample). With all 16 concepts this is the OLS MCE of
  `xai.refit_magnitudes`.

Reference: the mean replacement ẑ^(S) of `xai.concept_subsets` (FE_mean_replacement and
`mr_*`), the SAE's own linear path with the dropped concepts at their validation mean.
The logit probe is the least-squares optimum over all linear readouts of κ_S on
validation, so its validation MCE_subset is at most FE_mean_replacement there (asserted).

Everything is nucleus level (the explained model is a nucleus classifier): the
slide-level metrics of `xai.concept_subsets.Faithfulness` are not reported. Every AUPRC
column has the model's own test AUPRC next to it (`*_AUPRC_model`).

Signs: for every variant, subset and probe, whether each β_j has the sign of the SAE's
logit weight wᵀu_j (`coefficients.csv`); the concepts where it does not are listed per
subset in `bottleneck.csv` (`*_sign_mismatch`) and in `checks.json`.

Outputs in `out_bottleneck_dir`: `bottleneck.csv` (one row per variant and subset),
`coefficients.csv` (both probes' coefficients per concept, next to wᵀu_j),
`checks.json`, `figures/bottleneck.{pdf,png}`. Concepts are named by their review
letter if stable, by their SAE index (#j) otherwise.
"""

import json
import time
from pathlib import Path

import hydra
import matplotlib


matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from nuclei_graph.mlflow_utils import tag_parent_run
from nuclei_graph.xai.activations import load_activations
from nuclei_graph.xai.metrics import (
    LinearHead,
    baseline,
    classification_metrics,
    concept_logit_weights,
    fit_ols,
)
from nuclei_graph.xai.sae import load_autoencoder
from xai.concept_masks import code_commit
from xai.concept_morphometry import FIG_W_CM, THESIS_RC
from xai.concept_subsets import (
    COLORS,
    REFERENCE_COLOR,
    SPLITS,
    VARIANTS,
    Faithfulness,
    concept_names,
    fit_logistic,
    importance_ranking,
    mean_replaced,
)
from xai.evaluate_sae import head_logits


def nested_subsets(
    order: NDArray, stable: list[int]
) -> list[tuple[int, tuple[int, ...], int | None]]:
    """(m, S, the concept dropped to get from m + 1 to m) for m = |stable| … 1."""
    ranked = [int(j) for j in order if j in stable]
    out = []
    for m in range(len(ranked), 0, -1):
        dropped = ranked[m] if m < len(ranked) else None
        out.append((m, tuple(sorted(ranked[:m])), dropped))
    return out


def log_loss(labels: NDArray, scores: NDArray) -> float:
    """Mean binary cross-entropy of logits `scores`."""
    return float(np.mean(np.logaddexp(0.0, scores) - labels * scores))


def nucleus_metrics(
    metrics: dict[str, float], prefix: str, error_name: str, model_auprc: float
) -> dict[str, float]:
    """The nucleus-level metrics of `Faithfulness`, renamed.

    FE* becomes `error_name`*, the rest get `prefix`; the slide-level ones are dropped and
    the model's AUPRC is put after the AUPRC.
    """
    out = {}
    for key, value in metrics.items():
        if key.startswith("slide_"):
            continue
        if key.startswith("FE"):
            out[error_name + key[2:]] = value
        else:
            out[f"{prefix}_{key}"] = value
        if key == "AUPRC":
            out[f"{prefix}_AUPRC_model"] = model_auprc
    return out


def sign_mismatches(
    coef: NDArray, subset: tuple[int, ...], weights: NDArray
) -> list[int]:
    """Concepts j of `subset` whose probe coefficient has not the sign of wᵀu_j."""
    return [j for i, j in enumerate(subset) if np.sign(coef[i]) != np.sign(weights[j])]


class Probes:
    """Both probes and the mean-replacement reference for one variant."""

    def __init__(
        self,
        kappa: dict[str, NDArray],
        labels: dict[str, NDArray],
        f_validation: NDArray,
        weights: NDArray,
        b0: float,
        faithfulness: Faithfulness,
        model_auprc: float,
        device: torch.device,
    ) -> None:
        self.kappa = kappa
        self.model_auprc = model_auprc
        self.labels = labels
        self.f_validation = f_validation
        self.weights = weights
        self.mean_phi = weights * kappa["validation"].mean(0, dtype=np.float64)
        self.phi_test = np.asarray(kappa["test"], np.float64) * weights
        self.b0 = b0
        self.faithfulness = faithfulness
        self.device = device

    def columns(self, subset: tuple[int, ...]) -> dict[str, NDArray]:
        cols = list(subset)
        return {s: np.asarray(self.kappa[s][:, cols], np.float64) for s in SPLITS}

    def label_probe(
        self, x: dict[str, NDArray]
    ) -> tuple[dict[str, float], NDArray, float]:
        coef, intercept, info = fit_logistic(
            x["validation"], self.labels["validation"], self.device
        )
        scores = x["test"] @ coef + intercept
        y = self.labels["test"]
        nucleus = classification_metrics(y, scores)
        row = {
            "label_AUROC": nucleus["AUROC"],
            "label_AUPRC": nucleus["AUPRC"],
            "label_AUPRC_model": self.model_auprc,
            "label_log_loss": log_loss(y, scores),
            "label_log_loss_validation": info["log_loss"],
            "label_fit_iterations": info["iterations"],
            "label_fit_grad_max": info["grad_max"],
        }
        return row, coef, intercept

    def logit_probe(
        self, x: dict[str, NDArray]
    ) -> tuple[dict[str, float], NDArray, float]:
        eta = fit_ols(x["validation"], self.f_validation)
        fit_val = eta.predict(x["validation"])
        row = nucleus_metrics(
            self.faithfulness(eta.predict(x["test"])),
            "logit",
            "MCE_subset",
            self.model_auprc,
        )
        row["MCE_subset_validation"] = float(
            np.sqrt(np.mean((self.f_validation - fit_val) ** 2))
        )
        return row, eta.coef, eta.intercept

    def mean_replacement(self, subset: tuple[int, ...]) -> dict[str, float]:
        z_hat = mean_replaced(self.phi_test, self.mean_phi, self.b0, subset)
        row = nucleus_metrics(
            self.faithfulness(z_hat), "mr", "FE_mean_replacement", self.model_auprc
        )
        phi_val = np.asarray(self.kappa["validation"], np.float64) * self.weights
        z_val = mean_replaced(phi_val, self.mean_phi, self.b0, subset)
        row["FE_mean_replacement_validation"] = float(
            np.sqrt(np.mean((self.f_validation - z_val) ** 2))
        )
        return row


# --- figure ---


def figure(
    table: pd.DataFrame, auroc_f: float, ticks: list[tuple[int, str]]
) -> plt.Figure:
    """Label probe AUROC and logit probe MCE_subset (mean replacement FE) against |S|.

    The logit probe's AUROC is within about 0.001 of the label probe's; it is in the CSV.
    """
    side = FIG_W_CM / 2.54
    fig, (ax_auc, ax_fe) = plt.subplots(1, 2, figsize=(side, side * 0.42))
    nested = table[table["subset"] != "all"]
    ax_auc.axhline(auroc_f, color=REFERENCE_COLOR, lw=0.8, ls=":", zorder=1)
    for variant in VARIANTS:
        v = nested[nested["variant"] == variant].sort_values("m")
        c = COLORS[variant]
        ax_auc.plot(v["m"], v["label_AUROC"], color=c, lw=1.2, marker="o", ms=2.5)
        ax_fe.plot(v["m"], v["FE_mean_replacement"], color=c, lw=0.9, ls=":", zorder=2)
        ax_fe.plot(v["m"], v["MCE_subset"], color=c, lw=1.2, marker="o", ms=2.5)
    ax_auc.set_ylabel("nucleus AUROC (test)")
    ax_fe.set_ylabel(
        r"$\mathrm{MCE}_\mathrm{subset} = \mathrm{RMSE}(f, \hat f)$ (test)"
    )
    ax_fe.set_ylim(bottom=0)
    for ax in (ax_auc, ax_fe):
        ax.set_xticks([m for m, _ in ticks], [label for _, label in ticks])
        ax.tick_params(axis="x", labelsize=6)
        ax.invert_xaxis()
        ax.set_xlabel(r"$|S|$ (stable concepts kept; concept dropped)")
        ax.grid(True, color="#e6e6e6", lw=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    ax_auc.set_title("prediction task: label probe", fontsize=8, pad=3)
    ax_fe.set_title("logit reconstruction: logit probe", fontsize=8, pad=3)
    handles = [
        mpl.lines.Line2D([], [], color=COLORS["l1"], lw=1.2, marker="o", ms=2.5),
        mpl.lines.Line2D([], [], color=COLORS["refit"], lw=1.2, marker="o", ms=2.5),
        mpl.lines.Line2D([], [], color="#333333", lw=0.9, ls=":"),
        mpl.lines.Line2D([], [], color=REFERENCE_COLOR, lw=0.8, ls=":"),
    ]
    labels = [
        r"L1 codes $\kappa$",
        r"refit codes $\tilde\kappa$",
        r"$\mathrm{FE}_\mathrm{mean\ replacement}$ of $\hat z^{(S)}$ (reference)",
        f"model $f$ (AUROC {auroc_f:.4f})",
    ]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=7,
        handlelength=2.2,
        columnspacing=1.2,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.2, top=0.82, wspace=0.22)
    return fig


# --- main ---


def compute(config: DictConfig) -> tuple[Path, dict[str, object]]:
    torch.set_num_threads(config.num_threads)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device != "cpu" else "cpu"
    )
    out = Path(config.out_bottleneck_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    model, meta = load_autoencoder(config.sae_model)
    assert (meta["k"], meta["seed"]) == (16, 1), meta
    decoder = model.decoder_weight.detach().double().numpy()
    b_d = model.decoder_bias.detach().double().numpy()
    head = LinearHead.from_npz(str(Path(config.activations_dir) / "head.npz"))
    weights = concept_logit_weights(decoder, head)
    b0 = baseline(b_d, head)
    k = len(weights)
    names, stable = concept_names(config.deck_key, list(config.stable_concepts))
    assert stable == sorted(config.stable_indices), (stable, config.stable_indices)

    data = {s: load_activations(config.activations_dir, s) for s in SPLITS}
    labels = {s: data[s].labels for s in SPLITS}
    f = {s: head_logits(head, data[s]) for s in SPLITS}
    codes = {
        variant: {s: np.load(config.codes[variant][s]) for s in SPLITS}
        for variant in VARIANTS
    }
    for variant in codes:
        for s in SPLITS:
            assert codes[variant][s].shape == (len(data[s]), k), (variant, s)

    # Faithfulness needs the test slides; its slide-level metrics are not reported
    slide_ids = data["test"].index["slide_id"].astype(str).to_numpy()
    starts = np.flatnonzero(np.r_[True, slide_ids[1:] != slide_ids[:-1]])
    assert len(np.unique(slide_ids)) == len(starts), "test slides are not contiguous"
    is_carcinoma = pd.read_parquet(config.slides).set_index("slide_id")["is_carcinoma"]
    slide_labels = is_carcinoma.loc[slide_ids[starts]].to_numpy(np.int64)
    faithfulness = Faithfulness(
        f["test"], labels["test"], starts, slide_labels, config.flip_margin
    )
    model_nucleus = classification_metrics(labels["test"], f["test"])

    _, order = importance_ranking(codes[config.ranking_variant]["validation"], weights)
    nested = nested_subsets(order, stable)
    checks: dict[str, object] = {
        "device": str(device),
        "n_validation": len(data["validation"]),
        "n_test": len(data["test"]),
        "model_test_AUROC": model_nucleus["AUROC"],
        "model_test_AUPRC": model_nucleus["AUPRC"],
        "model_test_log_loss": log_loss(labels["test"], f["test"]),
        "ranking_variant": config.ranking_variant,
        "drop_order": [names[d] for _, _, d in nested if d is not None],
    }
    subsets = [("all", k, tuple(range(k)), None)] + [
        (f"stable{m}", m, s, d) for m, s, d in nested
    ]

    rows, coefficient_rows, mismatches = [], [], []
    for variant in VARIANTS:
        probes = Probes(
            codes[variant],
            labels,
            f["validation"],
            weights,
            b0,
            faithfulness,
            model_nucleus["AUPRC"],
            device,
        )
        for name, m, subset, dropped in subsets:
            start = time.time()
            x = probes.columns(subset)
            label_row, label_coef, label_intercept = probes.label_probe(x)
            logit_row, logit_coef, logit_intercept = probes.logit_probe(x)
            del x
            wrong = {
                "label": sign_mismatches(label_coef, subset, weights),
                "logit": sign_mismatches(logit_coef, subset, weights),
            }
            row = {
                "variant": variant,
                "subset": name,
                "m": m,
                "concepts": " ".join(names[j] for j in subset),
                "dropped": "" if dropped is None else names[dropped],
                **label_row,
                **logit_row,
                **probes.mean_replacement(subset),
                **{
                    f"{probe}_probe_sign_mismatch": " ".join(names[j] for j in js)
                    for probe, js in wrong.items()
                },
                **{
                    f"n_{probe}_probe_sign_mismatch": len(js)
                    for probe, js in wrong.items()
                },
            }
            rows.append(row)
            print(
                variant,
                name,
                f"{time.time() - start:.1f}s",
                {key: round(v, 4) for key, v in row.items() if isinstance(v, float)},
                flush=True,
            )
            for i, j in enumerate(subset):
                coefficient_rows.append(
                    {
                        "variant": variant,
                        "subset": name,
                        "concept": names[j],
                        "sae_index": j,
                        "label_probe_coef": float(label_coef[i]),
                        "logit_probe_coef": float(logit_coef[i]),
                        "sae_logit_weight": float(weights[j]),
                        "label_probe_sign_agrees": j not in wrong["label"],
                        "logit_probe_sign_agrees": j not in wrong["logit"],
                    }
                )
            for probe, js in wrong.items():
                coef = label_coef if probe == "label" else logit_coef
                for j in js:
                    mismatches.append(
                        {
                            "variant": variant,
                            "subset": name,
                            "probe": probe,
                            "concept": names[j],
                            "coef": float(coef[subset.index(j)]),
                            "sae_logit_weight": float(weights[j]),
                        }
                    )
            coefficient_rows.append(
                {
                    "variant": variant,
                    "subset": name,
                    "concept": "(intercept)",
                    "sae_index": -1,
                    "label_probe_coef": label_intercept,
                    "logit_probe_coef": logit_intercept,
                    "sae_logit_weight": b0,
                }
            )
        del probes

    table = pd.DataFrame(rows)
    coefficients = pd.DataFrame(coefficient_rows)

    # cross-checks with the earlier analyses
    stored = pd.read_csv(config.refit_metrics).query("split == 'test'")
    stored = stored.set_index("variant")
    for variant, key in (("l1", "sae_l1"), ("refit", "sae_l1_nnls_refit")):
        mine = table.query("variant == @variant and subset == 'all'").iloc[0]
        checks[f"{variant}/all16_MCE_subset_minus_stored_MCE"] = float(
            mine["MCE_subset"] - stored.loc[key, "MCE"]
        )
        checks[f"{variant}/all16_FE_mean_replacement_minus_stored_FE"] = float(
            mine["FE_mean_replacement"] - stored.loc[key, "FE"]
        )
    previous = pd.read_csv(config.subsets_table)
    same = {
        "stable8": "stable",
        "stable4": "top4",
        "stable3": "top3",
        "stable2": "top2",
        "stable1": "top1",
        "all": "all",
    }
    diffs = {}
    for variant in VARIANTS:
        for mine_name, theirs_name in same.items():
            mine = table.query("variant == @variant and subset == @mine_name").iloc[0]
            theirs = previous.query(
                "variant == @variant and subset == @theirs_name"
            ).iloc[0]
            assert sorted(mine["concepts"].split()) == sorted(
                theirs["concepts"].split()
            ), (mine_name, theirs_name)
            diffs[f"{variant}/{mine_name}"] = {
                "label_AUROC_minus_head_AUROC": float(
                    mine["label_AUROC"] - theirs["head_AUROC"]
                ),
                "FE_mean_replacement_minus_FE": float(
                    mine["FE_mean_replacement"] - theirs["FE"]
                ),
            }
    checks["vs_concept_subsets"] = diffs
    # OLS is optimal among linear readouts of κ_S on validation, ẑ^(S) is one of them
    excess = table["MCE_subset_validation"] - table["FE_mean_replacement_validation"]
    checks["max_MCE_subset_minus_FE_mean_replacement_validation"] = float(excess.max())
    assert excess.max() <= 1e-9, excess.max()
    checks["sign_mismatches"] = mismatches

    table.to_csv(out / "bottleneck.csv", index=False)
    coefficients.to_csv(out / "coefficients.csv", index=False)
    (out / "checks.json").write_text(json.dumps(checks, indent=2))
    ticks = [(m, f"{m}" if d is None else f"{m}\n$-${names[d]}") for m, _, d in nested]
    with mpl.rc_context(THESIS_RC):
        fig = figure(table, model_nucleus["AUROC"], ticks)
        fig.savefig(out / "figures" / "bottleneck.pdf")
        fig.savefig(out / "figures" / "bottleneck.png", dpi=300)
        plt.close(fig)
    print(json.dumps(checks, indent=2))
    return out, checks


@with_cli_args(["+xai=concept_bottleneck"])
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
    table = pd.read_csv(out / "bottleneck.csv")
    skip = ("variant", "subset", "concepts", "dropped", "m")
    skip += tuple(
        c
        for c in table.columns
        if c.endswith("_probe_sign_mismatch") and not c.startswith("n_")
    )
    for row in table.to_dict("records"):
        logger.log_metrics(
            {
                f"test/{row['variant']}/{row['subset']}/{key}": float(value)
                for key, value in row.items()
                if key not in skip and "_fit_" not in key and np.isfinite(float(value))
            }
        )
    for name in ("bottleneck.csv", "coefficients.csv", "checks.json"):
        logger.log_artifact(str(out / name))
    for name in ("bottleneck.pdf", "bottleneck.png"):
        logger.log_artifact(str(out / "figures" / name), "figures")
    for source in (__file__, "configs/xai/concept_bottleneck.yaml"):
        logger.log_artifact(str(Path(__file__).parents[1] / source), "code")


if __name__ == "__main__":
    main()
