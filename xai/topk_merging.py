"""The dominant TopK latents vs. the k = 16 concepts, and the matching figure (step 4, last items).

Merging check (`merging.csv`): for every seed of the undercomplete (m = 16, K = 4) and the
main (m = 1024, K = 16) TopK, the latent with the largest share of mean |φ| on test; its
decoder cosine (unit-norm directions) and the Pearson correlation of its activation with
κ_j over all test nuclei, for the concepts `merge_concepts` of the k = 16 SAE (seed 1);
and the Pearson correlation of its φ_l with Σ_j φ_j over each set in `merge_sums`.

Figure (`figures/k16_vs_topk_matching.{pdf,png}`): decoder cosine of the 16 concepts
(rows, stable eight first) with the top latents of one main seed (columns, by rank);
outlined cells are the Hungarian matches of the concepts to the whole dictionary
(signed cosine, as the stability matching) that fall among the shown latents. The last
column gives each concept's best activation correlation over all latents and its latent
(`matching.csv`, 2M random test nuclei).
"""

from pathlib import Path

import hydra
import matplotlib


matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
from numpy.typing import NDArray
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from scipy.optimize import linear_sum_assignment

from nuclei_graph.mlflow_utils import tag_parent_run
from nuclei_graph.xai.activations import load_activations
from nuclei_graph.xai.metrics import LinearHead, concept_logit_weights
from nuclei_graph.xai.sae import load_autoencoder
from nuclei_graph.xai.topk import encode_topk, load_topk
from xai.compare_topk_concepts import dense_columns, sparse_importance, unit_rows
from xai.concept_masks import code_commit
from xai.concept_morphometry import THESIS_RC


# diverging blue (similar) <-> gray <-> red (opposite)
COSINE_CMAP = LinearSegmentedColormap.from_list(
    "cosine", ["#e34948", "#f0efec", "#2a78d6"]
)
INK, MUTED = "#1f1f1e", "#6b6b68"


def pearson(x: NDArray, y: NDArray) -> float:
    return float(
        np.corrcoef(np.asarray(x, np.float64), np.asarray(y, np.float64))[0, 1]
    )


def merging_rows(
    model_name: str,
    seed: int,
    importance: pd.DataFrame,
    z: NDArray,
    direction: NDArray,
    kappa: NDArray,
    w_ref: NDArray,
    dirs_ref: NDArray,
    letters: pd.Series,
    config: DictConfig,
) -> list[dict]:
    top = importance.nsmallest(1, "rank").iloc[0]
    base = {
        "model": model_name,
        "seed": seed,
        "latent": int(top["latent"]),
        "share_abs_phi": float(top["share_abs_phi"]),
        "logit_weight": float(top["logit_weight"]),
        "frequency": float(top["frequency"]),
    }
    rows = [
        {
            **base,
            "concept": letters[j],
            "sae_index": j,
            "cosine": float(direction @ dirs_ref[j]),
            "correlation": pearson(z, kappa[:, j]),
        }
        for j in config.merge_concepts
    ]
    phi = z * top["logit_weight"]
    for group in config.merge_sums:
        summed = sum(kappa[:, j].astype(np.float64) * w_ref[j] for j in group)
        rows.append(
            {
                **base,
                "concept": "+".join(letters[j] for j in group),
                "sae_index": "+".join(map(str, group)),
                "cosine": np.nan,
                "correlation_phi": pearson(phi, summed),
            }
        )
    return rows


def matching_figure(
    cosine: NDArray,
    latents: list[int],
    hungarian: NDArray,
    concepts: pd.DataFrame,
    best: pd.DataFrame,
    width_cm: float,
) -> plt.Figure:
    """Rows: concepts in `concepts` order; columns: `latents` (by rank)."""
    rows, cols = len(concepts), len(latents)
    width = width_cm / 2.54
    cells, header, footer = 0.165 * rows, 0.62, 0.08  # inches
    height = cells + header + footer
    box = (footer / height, cells / height)  # bottom, height (figure fractions)
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0.12, box[0], 0.61, box[1]))
    values = cosine[np.ix_(concepts["concept"].to_numpy(), latents)]
    image = ax.imshow(values, cmap=COSINE_CMAP, vmin=-1, vmax=1, aspect="auto")
    for r in range(rows):
        for c in range(cols):
            v = values[r, c]
            ax.text(
                c,
                r,
                f"{v:.2f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color="white" if abs(v) > 0.6 else INK,
            )
    for r, j in enumerate(concepts["concept"]):
        if hungarian[j] in latents:
            c = latents.index(int(hungarian[j]))
            ax.add_patch(
                Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, lw=1.3, ec=INK)
            )
    n_stable = int(concepts["stable"].sum())
    ax.axhline(n_stable - 0.5, color="white", lw=2.5)
    ax.set_yticks(
        range(rows),
        [
            f"{l} ({j})"
            for l, j in zip(concepts["letter"], concepts["concept"], strict=True)
        ],
        fontsize=6.5,
    )
    ax.set_xticks(range(cols), [str(rank) for rank in range(1, cols + 1)], fontsize=6.5)
    ax.xaxis.tick_top()
    for c, latent in enumerate(latents):  # index in small print, between rank and cells
        ax.text(
            c, -0.58, f"L{latent}", ha="center", va="bottom", fontsize=4.5, color=MUTED
        )
    ax.set_title("latent rank (index)", loc="left", fontsize=6.5, pad=17)
    ax.tick_params(length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.012,
        n_stable / 2 - 0.5,
        "stable",
        rotation=90,
        ha="center",
        va="center",
        fontsize=6,
        color=MUTED,
        transform=blended_transform_factory(fig.transFigure, ax.transData),
    )

    side = fig.add_axes((0.74, box[0], 0.11, box[1]), sharey=ax)
    side.set_xlim(0, 1)
    for r, row in enumerate(best.itertuples(index=False)):
        side.text(
            0.08,
            r,
            f"{row.best_correlation:.2f}",
            ha="left",
            va="center",
            fontsize=5.5,
            color=INK,
        )
        side.text(
            0.62,
            r,
            f"L{row.best_correlation_latent}",
            ha="left",
            va="center",
            fontsize=4.5,
            color=MUTED,
        )
    side.axhline(n_stable - 0.5, color="#d0d0cc", lw=0.6)
    side.set_xticks([])
    side.tick_params(left=False, labelleft=False)
    for spine in side.spines.values():
        spine.set_visible(False)
    side.set_title("best $r$\n(latent)", fontsize=6.5, pad=8)

    bar = fig.add_axes((0.88, box[0] + 0.15 * box[1], 0.018, 0.7 * box[1]))
    colorbar = fig.colorbar(image, cax=bar, ticks=[-1, 0, 1])
    colorbar.ax.tick_params(labelsize=6, length=2)
    colorbar.outline.set_linewidth(0.4)
    colorbar.set_label("decoder cosine", fontsize=6.5)
    return fig


@with_cli_args(["+xai=topk_merging"])
@hydra.main(config_path="../configs", config_name="xai", version_base=None)
@autolog(
    log_config=False, log_stream=False
)  # attached to the TopK run: keep its configs/ and console.log
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)
    commit, dirty = code_commit("xai", "nuclei_graph", "configs")
    logger.experiment.set_tag(
        logger.run_id,
        "xai.merging_code_commit",
        f"{commit}{' (dirty)' if dirty else ''}",
    )
    torch.set_num_threads(config.num_threads)
    out = Path(config.output_dir)
    for source in (__file__, "configs/xai/topk_merging.yaml"):
        logger.log_artifact(str(Path(__file__).parents[1] / source), "code")

    head = LinearHead.from_npz(str(Path(config.activations_dir) / "head.npz"))
    test = load_activations(config.activations_dir, "test")
    key = pd.read_csv(config.deck_key)
    letters = key.set_index("concept")["letter"].astype(str)
    reference = load_autoencoder(config.reference_models[config.reference_seed])[0]
    dirs_ref = unit_rows(reference.decoder_weight)
    w_ref = concept_logit_weights(
        reference.decoder_weight.detach().double().numpy(), head
    )
    kappa = np.load(config.reference_codes)
    topk_dir = Path(config.topk_dir)

    rows = []
    for spec, stored in ((config.undercomplete, False), (config.main, True)):
        for s in spec.seeds:
            name = f"topk_m{spec.num_latents}_k{spec.k}_seed{s}"
            model = load_topk(str(topk_dir / "models" / f"{name}.pt"))[0]
            if stored:
                indices = np.load(topk_dir / "codes" / name / "codes_test_indices.npy")
                values = np.load(topk_dir / "codes" / name / "codes_test_values.npy")
            else:
                indices, values = encode_topk(model, test.h)
            U = model.decoder_weight.detach().double().numpy()
            importance = sparse_importance(
                indices, values, concept_logit_weights(U, head)
            )
            latent = int(importance.nsmallest(1, "rank")["latent"].iloc[0])
            z = dense_columns(indices, values, [latent])[:, 0]
            rows += merging_rows(
                name,
                s,
                importance,
                z,
                unit_rows(model.w_dec)[latent],
                kappa,
                w_ref,
                dirs_ref,
                letters,
                config,
            )
    merging = pd.DataFrame(rows)
    merging.to_csv(out / "merging.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(merging.round(3).to_string(index=False))

    # figure: the stable eight (by letter), then the others (by letter, the example last)
    f = config.figure
    name = f"topk_m{config.main.num_latents}_k{config.main.k}_seed{f.seed}"
    model = load_topk(str(topk_dir / "models" / f"{name}.pt"))[0]
    cosine = dirs_ref @ unit_rows(model.w_dec).T  # (16, m)
    _, hungarian = linear_sum_assignment(-cosine)  # rows 0..15 in order
    importance = pd.read_csv(out / "importance.csv").query("model == @name")
    latents = importance.nsmallest(f.top_latents, "rank")["latent"].astype(int).tolist()
    concepts = (
        key[["concept", "letter"]]
        .assign(
            stable=key["concept"].isin(list(config.stable_concepts)),
            example=key["letter"] == "Example",
        )
        .sort_values(["stable", "example", "letter"], ascending=[False, True, True])
    )
    best = (
        pd.read_csv(out / "matching.csv")
        .query("seed == @f.seed")
        .set_index("sae_index")
        .loc[concepts["concept"]]
    )
    with mpl.rc_context(THESIS_RC):
        fig = matching_figure(cosine, latents, hungarian, concepts, best, f.width_cm)
        fig.savefig(f"{f.path}.pdf")
        fig.savefig(f"{f.path}.png", dpi=300)
        plt.close(fig)
    outlined = sum(int(hungarian[j]) in latents for j in concepts["concept"])
    print(f"Hungarian matches among the top {f.top_latents}: {outlined}")
    logger.log_metrics({"merging/figure_outlined_matches": float(outlined)})
    logger.log_artifact(str(out / "merging.csv"), "comparison")
    for suffix in ("pdf", "png"):
        logger.log_artifact(f"{f.path}.{suffix}", "comparison/figures")


if __name__ == "__main__":
    main()
