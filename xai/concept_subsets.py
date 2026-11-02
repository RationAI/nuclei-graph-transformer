r"""Concept subsets of the k = 16 seed-1 SAE: how much of the model do some concepts carry?

Done separately for the encoder's codes κ (L1) and the NNLS refit κ̃ of
`xai.refit_magnitudes` (same support, unpenalised magnitudes); everything below is
computed for each variant with its own codes. No retraining.

Faithfulness (mean replacement). With φ_ij = κ_ij wᵀu_j and b₀ = wᵀb_D + β,

    ẑ_i^(S) = b₀ + Σ_{j∈S} φ_ij + Σ_{j∉S} φ̄_j,

φ̄_j the mean of φ_ij over all validation nuclei. ẑ^(all) is the SAE's reconstructed
logit f_A, ẑ^(∅) a constant. Reported on all test nuclei against the model's logit f
(= class_head(h), the z of the spec): FE = RMSE(f, ẑ), FE / std(f), R² = 1 − SSE / SST
of f, Spearman(f, ẑ), class flips sign(f) ≠ sign(ẑ) on all nuclei and among
|f| < `flip_margin`, AUROC / AUPRC of ẑ against the nucleus labels; slide level, the mean
of ẑ over each test slide against `is_carcinoma`: AUROC and ΔAUROC = AUROC(mean ẑ) −
AUROC(mean f).

Information (refit head). An unpenalised logistic regression of the nucleus label on the
activations (κ_j or κ̃_j) of S, fitted on all validation nuclei (Newton's method on the
standardised features, float64, step halving; checked against scikit-learn on a sample),
evaluated on test: AUROC / AUPRC. ∅ is the intercept-only head (AUROC 0.5).

Subsets: all 16; the stable eight (A–H); the unstable eight; top-m by the importance share
mean |φ_j| / Σ_k mean |φ_k| on validation for m in `top_m`; insertion (top-m kept) and
deletion (top-m mean-replaced) for m = 0…16. One ranking for both variants, that of
`ranking_variant` (the L1 codes); each variant's own ranking is reported in `ranking.csv`
only. Concepts are named by their review letter if stable, by their SAE index (#j)
otherwise.

Outputs in `out_dir`: `subsets.csv` (the named subsets), `curves.csv` (insertion and
deletion), `ranking.csv` (validation shares and ranks per variant), `checks.json`,
`figures/insertion_deletion.{pdf,png}` (nucleus AUROC and FE; AUPRC in the CSVs only).
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
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from nuclei_graph.mlflow_utils import tag_parent_run
from nuclei_graph.xai.activations import load_activations
from nuclei_graph.xai.metrics import (
    LinearHead,
    baseline,
    classification_metrics,
    concept_logit_weights,
)
from nuclei_graph.xai.sae import load_autoencoder
from xai.concept_masks import code_commit
from xai.concept_morphometry import FIG_W_CM, THESIS_RC, concept_letters
from xai.evaluate_sae import head_logits


SPLITS = ("validation", "test")
VARIANTS = {"l1": "L1 codes κ", "refit": "refit codes κ̃"}
COLORS = {"l1": "#2a78d6", "refit": "#eb6834"}
REFERENCE_COLOR = "#7a7a7a"


# --- subsets ---


def concept_names(deck_key: str, stable: list[str]) -> tuple[dict[int, str], list[int]]:
    """Display name per SAE index (letter if stable, #j otherwise) and the stable indices."""
    names, stable_idx = {}, []
    for j, letter in concept_letters(deck_key):
        if letter in stable:
            names[j] = letter
            stable_idx.append(j)
        else:
            names[j] = f"#{j}"
    assert len(stable_idx) == len(stable)
    return names, sorted(stable_idx)


def importance_ranking(kappa: NDArray, weights: NDArray) -> tuple[NDArray, NDArray]:
    """Share of mean |φ_j| (on the given split) and the concepts in descending order."""
    mean_abs_phi = np.abs(weights) * kappa.mean(0, dtype=np.float64)
    share = mean_abs_phi / mean_abs_phi.sum()
    return share, np.argsort(-share, kind="stable")


def named_subsets(
    order: NDArray, stable: list[int], top_m: list[int], k: int
) -> list[tuple[str, tuple[int, ...]]]:
    unstable = [j for j in range(k) if j not in stable]
    subsets = [
        ("all", tuple(range(k))),
        ("stable", tuple(stable)),
        ("unstable", tuple(unstable)),
    ]
    subsets += [(f"top{m}", tuple(sorted(order[:m].tolist()))) for m in top_m]
    return subsets


def curve_subsets(order: NDArray) -> list[tuple[str, int, tuple[int, ...]]]:
    k = len(order)
    out = []
    for m in range(k + 1):
        out.append(("insertion", m, tuple(sorted(order[:m].tolist()))))
        out.append(("deletion", m, tuple(sorted(order[m:].tolist()))))
    return out


# --- faithfulness ---


class Faithfulness:
    """Metrics of ẑ^(S) against f on test; the parts of f that do not depend on S are cached."""

    def __init__(
        self,
        f: NDArray,
        labels: NDArray,
        starts: NDArray,
        slide_labels: NDArray,
        flip_margin: float,
    ) -> None:
        self.f = f
        self.labels = labels
        self.starts = starts
        self.lengths = np.diff(np.append(starts, len(f)))
        self.slide_labels = slide_labels
        self.small = np.abs(f) < flip_margin
        self.rank_f = rankdata(f)
        self.sst = float(((f - f.mean()) ** 2).sum())
        self.std_f = float(f.std())
        self.slide_auroc_f = classification_metrics(slide_labels, self.slide_mean(f))[
            "AUROC"
        ]

    def slide_mean(self, values: NDArray) -> NDArray:
        return np.add.reduceat(values, self.starts) / self.lengths

    def __call__(self, z_hat: NDArray) -> dict[str, float]:
        f = self.f
        sse = float(((f - z_hat) ** 2).sum())
        fe = float(np.sqrt(sse / len(f)))
        constant = bool(np.ptp(z_hat) == 0)
        spearman = (
            np.nan
            if constant
            else float(np.corrcoef(self.rank_f, rankdata(z_hat))[0, 1])
        )
        flip = (f > 0) != (z_hat > 0)
        nucleus = classification_metrics(self.labels, z_hat)
        # a constant ẑ: per-slide sums would differ by rounding, which AUROC would rank
        slide_z = (
            np.full(len(self.starts), z_hat[0]) if constant else self.slide_mean(z_hat)
        )
        slide = classification_metrics(self.slide_labels, slide_z)
        return {
            "FE": fe,
            "FE_over_std_f": fe / self.std_f,
            "R2": 1 - sse / self.sst,
            "spearman_f": spearman,
            "flip_rate": float(flip.mean()),
            "flip_rate_small_logit": float(flip[self.small].mean()),
            "AUROC": nucleus["AUROC"],
            "AUPRC": nucleus["AUPRC"],
            "slide_mean_AUROC": slide["AUROC"],
            "slide_mean_dAUROC": slide["AUROC"] - self.slide_auroc_f,
        }


def mean_replaced(
    phi: NDArray, mean_phi: NDArray, b0: float, subset: tuple[int, ...]
) -> NDArray:
    out_of_subset = [j for j in range(phi.shape[1]) if j not in subset]
    constant = b0 + float(mean_phi[out_of_subset].sum())
    if not subset:
        return np.full(len(phi), constant)
    return phi[:, list(subset)].sum(1) + constant


# --- information ---


def fit_logistic(
    x: NDArray,
    y: NDArray,
    device: torch.device,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> tuple[NDArray, float, dict[str, float]]:
    """Unpenalised logistic regression by Newton's method with step halving.

    Features are standardised (validation mean and std) for conditioning; the returned
    coefficients are on the original scale. Converged when the max |gradient| of the mean
    log-loss falls below `tol`.
    """
    mu, sd = x.mean(0), x.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    xt = torch.from_numpy((x - mu) / sd).to(device, torch.float64)
    xt = torch.cat([torch.ones(len(xt), 1, dtype=xt.dtype, device=device), xt], 1)
    yt = torch.from_numpy(y.astype(np.float64)).to(device)
    n = len(yt)
    beta = torch.zeros(xt.shape[1], dtype=xt.dtype, device=device)
    prior = float(yt.mean())
    beta[0] = np.log(prior / (1 - prior))

    def loss(b: torch.Tensor) -> float:
        eta = xt @ b
        return float((torch.nn.functional.softplus(eta) - yt * eta).sum() / n)

    current = loss(beta)
    grad_max = np.inf
    for iteration in range(max_iter):
        eta = xt @ beta
        p = torch.sigmoid(eta)
        grad = xt.T @ (p - yt) / n
        grad_max = float(grad.abs().max())
        if grad_max < tol:
            break
        hess = (xt * (p * (1 - p))[:, None]).T @ xt / n
        step = torch.linalg.solve(hess, grad)
        t = 1.0
        while True:
            candidate = beta - t * step
            value = loss(candidate)
            if value <= current or t < 1e-8:
                break
            t /= 2
        beta, current = candidate, value
    b = beta.cpu().numpy()
    coef = b[1:] / sd
    intercept = float(b[0] - (mu / sd) @ b[1:])
    return (
        coef,
        intercept,
        {"iterations": iteration, "grad_max": grad_max, "log_loss": current},
    )


def information(
    kappa_val: NDArray,
    kappa_test: NDArray,
    y_val: NDArray,
    y_test: NDArray,
    subset: tuple[int, ...],
    device: torch.device,
) -> tuple[dict[str, float], dict[str, float]]:
    if not subset:
        scores = np.zeros(len(y_test))
        return {
            f"head_{k}": v for k, v in classification_metrics(y_test, scores).items()
        }, {}
    cols = list(subset)
    coef, intercept, info = fit_logistic(
        np.asarray(kappa_val[:, cols], np.float64), y_val, device
    )
    scores = np.asarray(kappa_test[:, cols], np.float64) @ coef + intercept
    metrics = classification_metrics(y_test, scores)
    return {f"head_{k}": v for k, v in metrics.items()}, info


def sklearn_check(
    kappa: NDArray, y: NDArray, device: torch.device, n: int, seed: int
) -> dict[str, float]:
    """Newton's solution against scikit-learn's (lbfgs, no penalty) on a random sample."""
    rows = np.sort(np.random.default_rng(seed).choice(len(y), n, replace=False))
    x, t = np.asarray(kappa[rows], np.float64), y[rows]
    coef, intercept, _ = fit_logistic(x, t, device)
    sd = x.std(0)
    reference = LogisticRegression(C=np.inf, max_iter=10000, tol=1e-12).fit(
        (x - x.mean(0)) / sd, t
    )
    ref_coef = reference.coef_[0] / sd
    eta_mine, eta_ref = (
        x @ coef + intercept,
        x @ ref_coef
        + (reference.intercept_[0] - (x.mean(0) / sd) @ reference.coef_[0]),
    )
    return {
        "n": n,
        "max_abs_coef_diff": float(np.abs(coef - ref_coef).max()),
        "max_abs_logit_diff": float(np.abs(eta_mine - eta_ref).max()),
    }


class SubsetEvaluator:
    """Faithfulness and information of one variant's subsets, each computed once."""

    def __init__(
        self,
        variant: str,
        phi: NDArray,
        mean_phi: NDArray,
        b0: float,
        kappa_val: NDArray,
        kappa_test: NDArray,
        labels: dict[str, NDArray],
        faithfulness: Faithfulness,
        device: torch.device,
        names: dict[int, str],
    ) -> None:
        self.variant = variant
        self.phi, self.mean_phi, self.b0 = phi, mean_phi, b0
        self.kappa_val, self.kappa_test = kappa_val, kappa_test
        self.labels = labels
        self.faithfulness = faithfulness
        self.device = device
        self.names = names
        self.cache: dict[tuple[int, ...], dict[str, float]] = {}

    def __call__(self, subset: tuple[int, ...]) -> dict[str, float]:
        if subset not in self.cache:
            start = time.time()
            z_hat = mean_replaced(self.phi, self.mean_phi, self.b0, subset)
            row = self.faithfulness(z_hat)
            head_metrics, fit_info = information(
                self.kappa_val,
                self.kappa_test,
                self.labels["validation"],
                self.labels["test"],
                subset,
                self.device,
            )
            row |= head_metrics
            row |= {f"head_fit_{key}": v for key, v in fit_info.items()}
            self.cache[subset] = row
            print(
                self.variant,
                [self.names[j] for j in subset],
                f"{time.time() - start:.1f}s",
                {key: round(v, 4) for key, v in row.items()},
                flush=True,
            )
        return self.cache[subset]


# --- figure ---


def figure(curves: pd.DataFrame, auroc_f: float, order_names: list[str]) -> plt.Figure:
    """Insertion and deletion: nucleus AUROC (mean replacement and refit head) and FE."""
    side = FIG_W_CM / 2.54
    fig, axes = plt.subplots(2, 2, figsize=(side, side * 0.66), sharex=True)
    curves = curves[curves["m"] >= 1]
    for col, kind in enumerate(("insertion", "deletion")):
        part = curves[curves["curve"] == kind]
        ax_auc, ax_fe = axes[0, col], axes[1, col]
        ax_auc.axhline(auroc_f, color=REFERENCE_COLOR, lw=0.8, ls=":", zorder=1)
        for variant in VARIANTS:
            v = part[part["variant"] == variant].sort_values("m")
            c = COLORS[variant]
            ax_auc.plot(v["m"], v["AUROC"], color=c, lw=1.2, marker="o", ms=2.5)
            ax_auc.plot(v["m"], v["head_AUROC"], color=c, lw=1.0, ls="--")
            ax_fe.plot(v["m"], v["FE"], color=c, lw=1.2, marker="o", ms=2.5)
        title = (
            "insertion: top $m$ kept, rest mean-replaced"
            if kind == "insertion"
            else "deletion: top $m$ mean-replaced"
        )
        ax_auc.set_title(title, fontsize=8, pad=3)
        verb = "added" if kind == "insertion" else "removed"
        ax_fe.set_xlabel(f"$m$ (concept {verb} at $m$; L1 validation ranking)")
        ax_fe.set_xticks(
            range(1, len(order_names) + 1),
            [f"{m}\n{name}" for m, name in enumerate(order_names, 1)],
        )
        ax_fe.tick_params(axis="x", labelsize=6)
        ax_fe.set_ylim(bottom=0)
    axes[0, 0].sharey(axes[0, 1])
    axes[1, 0].sharey(axes[1, 1])
    axes[0, 0].set_ylabel("nucleus AUROC (test)")
    axes[1, 0].set_ylabel(r"FE $=$ RMSE$(f, \hat z^{(S)})$")
    for ax in axes.flat:
        ax.grid(True, color="#e6e6e6", lw=0.5)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    handles = [
        mpl.lines.Line2D([], [], color=COLORS["l1"], lw=1.2, marker="o", ms=2.5),
        mpl.lines.Line2D([], [], color=COLORS["refit"], lw=1.2, marker="o", ms=2.5),
        mpl.lines.Line2D([], [], color="#333333", lw=1.2),
        mpl.lines.Line2D([], [], color="#333333", lw=1.0, ls="--"),
        mpl.lines.Line2D([], [], color=REFERENCE_COLOR, lw=0.8, ls=":"),
    ]
    labels = [
        r"L1 codes $\kappa$",
        r"refit codes $\tilde\kappa$",
        r"mean replacement $\hat z^{(S)}$",
        "refit logistic head on $S$",
        f"model $f$ (AUROC {auroc_f:.3f})",
    ]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=7,
        handlelength=2.2,
        columnspacing=1.2,
        bbox_to_anchor=(0.5, 1.0),
    )
    fig.subplots_adjust(
        left=0.08, right=0.99, bottom=0.14, top=0.88, wspace=0.12, hspace=0.12
    )
    return fig


# --- main ---


def compute(config: DictConfig) -> tuple[Path, dict[str, object]]:
    torch.set_num_threads(config.num_threads)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and config.device != "cpu" else "cpu"
    )
    out = Path(config.out_subsets_dir)
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
    f_test = head_logits(head, data["test"])
    codes = {
        "l1": {s: np.load(config.codes.l1[s]) for s in SPLITS},
        "refit": {s: np.load(config.codes.refit[s]) for s in SPLITS},
    }
    for variant in codes:
        for s in SPLITS:
            assert codes[variant][s].shape == (len(data[s]), k), (variant, s)

    slide_ids = data["test"].index["slide_id"].astype(str).to_numpy()
    starts = np.flatnonzero(np.r_[True, slide_ids[1:] != slide_ids[:-1]])
    assert len(np.unique(slide_ids)) == len(starts), "test slides are not contiguous"
    is_carcinoma = pd.read_parquet(config.slides).set_index("slide_id")["is_carcinoma"]
    slide_labels = is_carcinoma.loc[slide_ids[starts]].to_numpy(np.int64)
    faithfulness = Faithfulness(
        f_test, labels["test"], starts, slide_labels, config.flip_margin
    )
    model_nucleus = classification_metrics(labels["test"], f_test)
    checks: dict[str, object] = {
        "device": str(device),
        "n_validation": len(data["validation"]),
        "n_test": len(data["test"]),
        "n_test_slides": len(starts),
        "test_carcinoma_slides": int(slide_labels.sum()),
        "model_test_AUROC": model_nucleus["AUROC"],
        "model_test_AUPRC": model_nucleus["AUPRC"],
        "model_test_slide_mean_AUROC": faithfulness.slide_auroc_f,
        "max_abs_head_logit_vs_stored_test": float(
            np.abs(f_test - data["test"].logits).max()
        ),
    }

    # one ranking for both variants (top-m subsets and curves); each variant's own
    # validation ranking is only reported in ranking.csv
    _, order = importance_ranking(codes[config.ranking_variant]["validation"], weights)
    checks["ranking_variant"] = config.ranking_variant
    checks["ranking"] = [names[j] for j in order]
    checks["top8_equals_stable"] = sorted(order[:8].tolist()) == stable

    named_rows, curve_rows, ranking_rows = [], [], []
    for variant in VARIANTS:
        val, test = codes[variant]["validation"], codes[variant]["test"]
        share, own_order = importance_ranking(val, weights)
        mean_phi = weights * val.mean(0, dtype=np.float64)
        phi = np.asarray(test, np.float64) * weights
        for rank, j in enumerate(own_order, 1):
            ranking_rows.append(
                {
                    "variant": variant,
                    "rank": rank,
                    "concept": names[j],
                    "sae_index": int(j),
                    "stable": j in stable,
                    "logit_weight": float(weights[j]),
                    "share_abs_phi_validation": float(share[j]),
                    "mean_phi_validation": float(mean_phi[j]),
                    "used_for_subsets": variant == config.ranking_variant,
                }
            )
        checks[f"{variant}/own_ranking_equals_common"] = bool(
            (own_order == order).all()
        )
        evaluate = SubsetEvaluator(
            variant, phi, mean_phi, b0, val, test, labels, faithfulness, device, names
        )
        for name, subset in named_subsets(order, stable, list(config.top_m), k):
            named_rows.append(
                {
                    "variant": variant,
                    "subset": name,
                    "m": len(subset),
                    "concepts": " ".join(names[j] for j in subset),
                    **evaluate(subset),
                }
            )
        for kind, m, subset in curve_subsets(order):
            curve_rows.append(
                {
                    "variant": variant,
                    "curve": kind,
                    "m": m,
                    "concepts_kept": " ".join(names[j] for j in subset),
                    **evaluate(subset),
                }
            )
        checks[f"{variant}/sklearn_check_all16"] = sklearn_check(
            val, labels["validation"], device, config.check_n, config.seed
        )
        z_empty = mean_replaced(phi[:1], mean_phi, b0, ())[0]
        f_a_val_mean = b0 + float(mean_phi.sum())
        checks[f"{variant}/empty_subset_constant_minus_mean_fA_validation"] = float(
            z_empty - f_a_val_mean
        )
        del phi, evaluate

    named = pd.DataFrame(named_rows)
    curves = pd.DataFrame(curve_rows)
    ranking = pd.DataFrame(ranking_rows)

    # the full subset is the SAE's reconstruction: compare with xai.refit_magnitudes
    stored = pd.read_csv(config.refit_metrics).query("split == 'test'")
    stored = stored.set_index("variant")
    for variant, key in (("l1", "sae_l1"), ("refit", "sae_l1_nnls_refit")):
        mine = named.query("variant == @variant and subset == 'all'").iloc[0]
        checks[f"{variant}/all16_vs_refit_magnitudes"] = {
            c: {"stored": float(stored.loc[key, s]), "here": float(mine[c])}
            for c, s in (
                ("FE", "FE"),
                ("FE_over_std_f", "FE_over_std_f"),
                ("spearman_f", "spearman_f_fA"),
                ("AUROC", "AUROC_fA"),
                ("AUPRC", "AUPRC_fA"),
                ("flip_rate", "flip_rate"),
                ("flip_rate_small_logit", "flip_rate_small_logit"),
            )
        }
    importance = pd.read_csv(config.refit_importance).query("split == 'validation'")
    importance = importance.set_index("concept")
    for variant, column in (
        ("l1", "share_abs_phi_l1"),
        ("refit", "share_abs_phi_refit"),
    ):
        r = ranking.query("variant == @variant").set_index("sae_index")
        checks[f"{variant}/share_validation_max_abs_diff_vs_refit_magnitudes"] = float(
            (r["share_abs_phi_validation"] - importance.loc[r.index, column])
            .abs()
            .max()
        )

    named.to_csv(out / "subsets.csv", index=False)
    curves.to_csv(out / "curves.csv", index=False)
    ranking.to_csv(out / "ranking.csv", index=False)
    (out / "checks.json").write_text(json.dumps(checks, indent=2))
    with mpl.rc_context(THESIS_RC):
        fig = figure(curves, model_nucleus["AUROC"], [names[j] for j in order])
        fig.savefig(out / "figures" / "insertion_deletion.pdf")
        fig.savefig(out / "figures" / "insertion_deletion.png", dpi=300)
        plt.close(fig)
    print(json.dumps(checks, indent=2))
    return out, checks


@with_cli_args(["+xai=concept_subsets"])
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
    named = pd.read_csv(out / "subsets.csv")
    for row in named.to_dict("records"):
        logger.log_metrics(
            {
                f"test/{row['variant']}/{row['subset']}/{key}": float(value)
                for key, value in row.items()
                if key not in ("variant", "subset", "concepts", "m")
                and not key.startswith("head_fit_")
                and np.isfinite(float(value))
            }
        )
    for name in ("subsets.csv", "curves.csv", "ranking.csv", "checks.json"):
        logger.log_artifact(str(out / name))
    for name in ("insertion_deletion.pdf", "insertion_deletion.png"):
        logger.log_artifact(str(out / "figures" / name), "figures")
    for source in (__file__, "configs/xai/concept_subsets.yaml"):
        logger.log_artifact(str(Path(__file__).parents[1] / source), "code")


if __name__ == "__main__":
    main()
