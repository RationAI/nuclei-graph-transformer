"""Galleries and concept maps of the most important TopK latents (step 4e).

For one main-configuration seed (`latent_seed`, default: the lowest holdout RE, the
rule that chose the k = 16 SAE's seed), the `top_latents` latents by mean |φ| on test,
rendered exactly as the review material (`xai.concept_material`: same regions, tissue
rule, color scale and gallery settings). The images name latents by their index only
(no matched concept), so they can later be blinded like the review deck.

Outputs in `output_dir/<model>/`: `latent_summary.csv`, `top_nuclei.parquet`,
`regions.parquet`, `region_rejections.csv`, `concept_maps/latent_<l>.png`,
`gallery/latent_<l>.png`.
"""

from pathlib import Path

import hydra
import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from omegaconf import DictConfig
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger

from nuclei_graph.mlflow_utils import tag_parent_run
from nuclei_graph.xai.activations import load_activations
from nuclei_graph.xai.metrics import LinearHead, concept_logit_weights
from nuclei_graph.xai.topk import load_topk
from xai.compare_topk_concepts import dense_columns, sparse_importance
from xai.concept_masks import code_commit
from xai.concept_material import (
    SlideReader,
    region_candidates,
    render_concept_map,
    render_gallery,
    select_regions,
    slide_tissue,
    top_nuclei,
)


@with_cli_args(["+xai=topk_latent_material"])
@hydra.main(config_path="../configs", config_name="xai", version_base=None)
@autolog(
    log_config=False, log_stream=False
)  # attached to the TopK run: keep its configs/ and console.log
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)
    commit, dirty = code_commit("xai", "nuclei_graph", "configs")
    logger.experiment.set_tag(
        logger.run_id,
        "xai.latent_material_code_commit",
        f"{commit}{' (dirty)' if dirty else ''}",
    )
    torch.set_num_threads(config.num_threads)
    topk_dir = Path(config.topk_dir)
    grid = pd.read_csv(topk_dir / "training_grid.csv")
    main_rows = grid[
        (grid["num_latents"] == config.main.num_latents) & (grid["k"] == config.main.k)
    ]
    seed = config.latent_seed
    if seed is None:
        seed = int(main_rows.loc[main_rows["RE"].idxmin(), "seed"])
    name = f"topk_m{config.main.num_latents}_k{config.main.k}_seed{seed}"
    out_dir = Path(config.output_dir) / name
    (out_dir / "concept_maps").mkdir(parents=True, exist_ok=True)
    (out_dir / "gallery").mkdir(exist_ok=True)
    for source in (__file__, "configs/xai/topk_latent_material.yaml"):
        logger.log_artifact(str(Path(__file__).parents[1] / source), "code")

    head = LinearHead.from_npz(str(Path(config.activations_dir) / "head.npz"))
    test = load_activations(config.activations_dir, "test")
    slides = pd.read_parquet(config.slides)
    slides = slides[slides["split"] == "test"].sort_values("offset")
    model = load_topk(str(topk_dir / "models" / f"{name}.pt"))[0]
    indices = np.load(topk_dir / "codes" / name / "codes_test_indices.npy")
    values = np.load(topk_dir / "codes" / name / "codes_test_values.npy")
    U = model.decoder_weight.detach().double().numpy()
    importance = sparse_importance(indices, values, concept_logit_weights(U, head))
    top = importance.nsmallest(config.top_latents, "rank").reset_index(drop=True)
    latents = top["latent"].tolist()
    codes = dense_columns(indices, values, latents)  # column j = latent latents[j]
    del indices, values

    labels = test.labels
    active = codes > 0
    top["carcinoma_share_active"] = [
        labels[active[:, j]].mean() for j in range(len(latents))
    ]
    top["color_scale_max"] = [
        np.quantile(c[c > 0], config.regions.vmax_quantile) for c in codes.T
    ]
    top.insert(0, "model", name)
    top.to_csv(out_dir / "latent_summary.csv", index=False)

    x = test.index["centroid_x_um"].to_numpy()
    y = test.index["centroid_y_um"].to_numpy()
    rows = [
        slice(int(slide.offset), int(slide.offset + slide.n_nuclei))
        for slide in slides.itertuples(index=False)
    ]
    tissue = Parallel(n_jobs=config.tissue.n_jobs)(
        delayed(slide_tissue)(
            str(Path(config.tissue.masks) / f"{slide.slide_id}.tiff"),
            float(slide.mpp_x),
            x[sl],
            y[sl],
            config.tissue.target_um_per_px,
            config.regions.stride_um,
        )
        for slide, sl in zip(slides.itertuples(index=False), rows, strict=True)
    )
    cell_tissue = {
        slide_id: cells
        for slide_id, (_, cells) in zip(slides["slide_id"], tissue, strict=True)
    }

    nuclei = top_nuclei(codes, test, config.top_n, config.max_per_slide)
    nuclei.insert(1, "latent", np.asarray(latents)[nuclei["concept"]])
    nuclei.to_parquet(out_dir / "top_nuclei.parquet", index=False)
    candidates, rejections = region_candidates(
        test, slides, codes, cell_tissue, config.regions
    )
    rejections.insert(1, "latent", np.asarray(latents)[rejections["concept"]])
    rejections.to_csv(out_dir / "region_rejections.csv", index=False)
    regions = select_regions(candidates, config.regions.num_high)
    regions.insert(1, "latent", np.asarray(latents)[regions["concept"]])
    regions.to_parquet(out_dir / "regions.parquet", index=False)

    reader = SlideReader(slides)
    for j, row in top.iterrows():
        latent = int(row["latent"])
        summary = pd.Series(
            {
                "logit_weight": row["logit_weight"],
                "share_abs_phi": row["share_abs_phi"],
                "activation_frequency": row["frequency"],
            }
        )
        latent_regions = regions[regions["concept"] == j]
        if len(latent_regions):
            fig = render_concept_map(
                j,
                latent_regions,
                reader,
                test,
                slides,
                codes,
                summary,
                row["color_scale_max"],
                config,
            )
            fig.suptitle(  # the latent index instead of the column of `codes`
                f"Latent {latent} (rank {row['rank']}) · wᵀu={row['logit_weight']:+.3f} · "
                f"share |φ|={row['share_abs_phi']:.1%} · freq={row['frequency']:.1%} · "
                f"{config.regions.size_um:g} µm regions",
                fontsize=11,
            )
            fig.axes[-1].set_xlabel(
                f"z_{latent} (inactive = gray; clipped at q{config.regions.vmax_quantile})"
            )
            fig.savefig(out_dir / "concept_maps" / f"latent_{latent:04d}.png", dpi=100)
            plt.close(fig)
        latent_top = nuclei[nuclei["concept"] == j].nsmallest(
            config.gallery.num, "rank"
        )
        if len(latent_top):
            render_gallery(latent_top, reader, test, slides, config.gallery).save(
                out_dir / "gallery" / f"latent_{latent:04d}.png"
            )
        print(f"latent {latent} (rank {row['rank']}): {len(latent_regions)} regions")

    fallback = regions.groupby("latent")["fallback"].sum()
    logger.log_metrics({"latent_material/fallback_regions": float(fallback.sum())})
    for path in out_dir.iterdir():
        if path.is_dir():
            logger.log_artifacts(
                str(path), f"comparison/latent_material/{name}/{path.name}"
            )
        else:
            logger.log_artifact(str(path), f"comparison/latent_material/{name}")


if __name__ == "__main__":
    main()
