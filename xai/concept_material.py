"""Material for interpreting the concepts of one SAE (no naming of concepts).

On all test nuclei it writes:
- `concept_summary.csv`, one row per concept;
- `regions.parquet` and `concept_maps/concept_<j>.png`: per concept, `regions.num_high`
  square regions with high mean activation from different slides (the best
  carcinoma-majority and benign-majority regions first) and one region where the
  concept is absent, rendered as H&E next to the nucleus polygons filled by the
  concept activation (shared color scale per concept, inactive = light gray);
- `top_nuclei.parquet` and `gallery/concept_<j>.png`: the highest-activating nuclei
  (at most `max_per_slide` per slide) with a small context view.
"""

from functools import lru_cache
from pathlib import Path

import hydra
import matplotlib
from joblib import Parallel, delayed
from scipy.stats import spearmanr


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize, to_rgba
from omegaconf import DictConfig
from PIL import Image, ImageDraw
from rationai.mlkit import autolog, with_cli_args
from rationai.mlkit.lightning.loggers import MLFlowLogger
from ratiopath.openslide import OpenSlide

from nuclei_graph.data.datasets.crop.prediction import PredictionDataset
from nuclei_graph.mlflow_utils import tag_parent_run
from nuclei_graph.nuclei_graph_typing import Box, SlideSize
from nuclei_graph.xai.activations import Activations, load_activations
from nuclei_graph.xai.metrics import LinearHead, concept_logit_weights
from nuclei_graph.xai.runtime import setup_precision
from nuclei_graph.xai.sae import ConceptAutoencoder, encode, load_autoencoder
from nuclei_graph.xai.stability import matched_cosines
from nuclei_graph.xai.tissue import TissueMask


def select_model(config: DictConfig) -> tuple[ConceptAutoencoder, dict]:
    """The selected SAE with `num_concepts`.

    Unless a seed is given, the seed with the lowest holdout (validation) RE.
    """
    models = pd.read_csv(Path(config.sae_dir) / "models.csv")
    candidates = models[
        (models["method"] == "sae") & (models["k"] == config.num_concepts)
    ]
    if config.sae_seed is not None:
        candidates = candidates[candidates["seed"] == config.sae_seed]
    assert len(candidates), f"No selected SAE with k={config.num_concepts}"
    best = candidates.loc[candidates["RE"].idxmin()]
    model, meta = load_autoencoder(str(Path(config.sae_dir) / best["path"]))
    return model, {**meta, "path": best["path"]}


def seed_stability(
    config: DictConfig, meta: dict, decoder_weight: np.ndarray
) -> pd.DataFrame:
    """Per concept: Hungarian-matched decoder cosine to the other selected seeds."""
    models = pd.read_csv(Path(config.sae_dir) / "models.csv")
    others = models[
        (models["method"] == "sae")
        & (models["k"] == meta["k"])
        & (models["seed"] != meta["seed"])
    ]
    stability = pd.DataFrame(index=np.arange(len(decoder_weight)))
    for row in others.sort_values("seed").itertuples(index=False):
        other = load_autoencoder(str(Path(config.sae_dir) / row.path))[0]
        match, cosine = matched_cosines(
            decoder_weight, other.decoder_weight.detach().numpy()
        )
        stability[f"matched_concept_seed{row.seed}"] = match
        stability[f"best_match_cosine_seed{row.seed}"] = cosine
    cosines = stability.filter(like="best_match_cosine")
    stability["unstable"] = (cosines < config.stability_threshold).any(axis=1)
    return stability


def concept_summary(
    codes: np.ndarray,
    labels: np.ndarray,
    decoder_weight: np.ndarray,
    head: LinearHead,
    f: np.ndarray,
    near_boundary: np.ndarray,
) -> pd.DataFrame:
    """One row per concept, over all test nuclei.

    `near_boundary` marks nuclei within `boundary_um` of the tissue-mask boundary.
    """
    logit_weights = concept_logit_weights(decoder_weight, head)
    # φ_j = (wᵀu_j) c_j, so mean |φ_j| = |wᵀu_j| · mean |c_j|
    mean_abs_phi = np.abs(logit_weights) * np.abs(codes).mean(0, dtype=np.float64)
    active = codes > 0
    total_c = codes.sum(0, dtype=np.float64)
    return pd.DataFrame(
        {
            "concept": np.arange(codes.shape[1]),
            "activation_frequency": active.mean(0),
            "mean_c_carcinoma": codes[labels == 1].mean(0, dtype=np.float64),
            "mean_c_benign": codes[labels == 0].mean(0, dtype=np.float64),
            "logit_weight": logit_weights,
            "logit_weight_sign": np.sign(logit_weights).astype(np.int64),
            "cosine_u_w": logit_weights
            / (np.linalg.norm(decoder_weight, axis=1) * np.linalg.norm(head.w)),
            "mean_abs_phi": mean_abs_phi,
            "share_abs_phi": mean_abs_phi / mean_abs_phi.sum(),
            "carcinoma_share_active": [
                labels[active[:, j]].mean() if active[:, j].any() else np.nan
                for j in range(codes.shape[1])
            ],
            "spearman_c_f": [
                spearmanr(codes[:, j], f).statistic for j in range(codes.shape[1])
            ],
            "boundary_activation_share": codes[near_boundary].sum(0, dtype=np.float64)
            / np.where(total_c > 0, total_c, np.nan),
        }
    )


def slide_tissue(
    mask_path: str,
    mpp: float,
    x: np.ndarray,
    y: np.ndarray,
    target_um_per_px: float,
    cell_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Boundary distance of the slide's nuclei (µm) and its cell tissue fractions."""
    mask = TissueMask.read(mask_path, mpp, target_um_per_px)
    return mask.boundary_distance_um(x, y), mask.cell_fraction(cell_um)


def top_nuclei(
    codes: np.ndarray, test: Activations, top_n: int, max_per_slide: int
) -> pd.DataFrame:
    slide_ids = test.index["slide_id"].astype(str).to_numpy()
    rows = []
    for j in range(codes.shape[1]):
        # enough candidates to fill top_n under the per-slide limit
        start = len(rows)
        num = min(len(codes), top_n * max_per_slide * 50)
        candidates = np.argpartition(-codes[:, j], num - 1)[:num]
        candidates = candidates[np.argsort(-codes[candidates, j], kind="stable")]
        per_slide: dict[str, int] = {}
        for i in candidates:
            if codes[i, j] <= 0 or len(rows) - start >= top_n:
                break
            if per_slide.get(slide_ids[i], 0) >= max_per_slide:
                continue
            per_slide[slide_ids[i]] = per_slide.get(slide_ids[i], 0) + 1
            rows.append({"concept": j, "rank": len(rows) - start, "row": i})

    top = pd.DataFrame(rows, columns=["concept", "rank", "row"])
    index = test.index.iloc[top["row"]].reset_index(drop=True)
    return pd.DataFrame(
        {
            "concept": top["concept"].to_numpy(),
            "rank": top["rank"].to_numpy(),
            "row": top["row"].to_numpy(),
            "slide_id": index["slide_id"].astype(str),
            "nucleus_id": index["nucleus_id"],
            "centroid_x_um": index["centroid_x_um"],
            "centroid_y_um": index["centroid_y_um"],
            "c": codes[top["row"], top["concept"]],
            "label": index["label"],
        }
    )


def box_sum(grid: np.ndarray, m: int) -> np.ndarray:
    """Sums over all mxm windows of cells (first two axes)."""
    padded = np.pad(
        grid.cumsum(0).cumsum(1), [(1, 0), (1, 0)] + [(0, 0)] * (grid.ndim - 2)
    )
    return padded[m:, m:] - padded[:-m, m:] - padded[m:, :-m] + padded[:-m, :-m]


def slide_windows(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    codes: np.ndarray,
    cell_tissue: np.ndarray,
    regions: DictConfig,
) -> pd.DataFrame:
    """Statistics of all size_um x size_um windows on a stride_um grid of one slide.

    `cell_tissue` is the tissue fraction of the stride_um cells of the slide
    (`TissueMask.cell_fraction`), indexed from the slide origin.
    """
    m = round(regions.size_um / regions.stride_um)
    ix = np.floor(x / regions.stride_um).astype(np.int64)
    iy = np.floor(y / regions.stride_um).astype(np.int64)
    ix0, iy0 = ix.min(), iy.min()
    ix, iy = ix - ix0, iy - iy0
    shape = (ix.max() + m, iy.max() + m)
    k = codes.shape[1]

    count = np.zeros(shape)
    np.add.at(count, (ix, iy), 1)
    carcinoma = np.zeros(shape)
    np.add.at(carcinoma, (ix, iy), labels)
    sum_c = np.zeros((*shape, k))
    np.add.at(sum_c, (ix, iy), codes)
    active = np.zeros((*shape, k))
    np.add.at(active, (ix, iy), codes > 0)

    # tissue fraction of the cells, aligned with the window grid (outside = 0)
    tissue = np.zeros(shape)
    tx = min(shape[0], max(0, cell_tissue.shape[0] - ix0))
    ty = min(shape[1], max(0, cell_tissue.shape[1] - iy0))
    tissue[:tx, :ty] = cell_tissue[ix0 : ix0 + tx, iy0 : iy0 + ty]

    count, carcinoma = box_sum(count, m), box_sum(carcinoma, m)
    sum_c, active = box_sum(sum_c, m), box_sum(active, m)
    tissue = box_sum(tissue, m) / m**2
    wx, wy = np.nonzero(count > 0)

    n = count[wx, wy]
    windows = pd.DataFrame(
        {
            "x0_um": (wx + ix0) * regions.stride_um,
            "y0_um": (wy + iy0) * regions.stride_um,
            "n_nuclei": n.astype(np.int64),
            "dense": n >= regions.min_nuclei,
            "tissue_fraction": tissue[wx, wy],
            "frac_carcinoma": carcinoma[wx, wy] / n,
        }
    )
    windows["mean_c"] = list(sum_c[wx, wy] / n[:, None])
    windows["frac_active"] = list(active[wx, wy] / n[:, None])
    return windows


def region_candidates(
    test: Activations,
    slides: pd.DataFrame,
    codes: np.ndarray,
    cell_tissue: dict[str, np.ndarray],
    regions: DictConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Candidate regions per slide and concept, and rejection counts per concept.

    Among windows with ≥ `min_nuclei` nuclei and a tissue fraction ≥
    `min_tissue_fraction`: the best carcinoma- and benign-majority windows;
    among windows with ≥ `min_nuclei` nuclei: the window with the lowest mean
    activation (most nuclei on ties); as a fallback for high regions, the densest
    window with positive mean activation that fails either rule.

    Also counts, per concept, the windows with positive mean activation that each
    rule rejected (the tissue rule counted among windows passing the nuclei rule).
    """
    labels = test.labels
    x = test.index["centroid_x_um"].to_numpy()
    y = test.index["centroid_y_um"].to_numpy()
    num_concepts = codes.shape[1]
    rejected = np.zeros(num_concepts, dtype=np.int64)
    rejected_tissue = np.zeros(num_concepts, dtype=np.int64)
    considered = np.zeros(num_concepts, dtype=np.int64)
    rows = []
    for slide in slides.itertuples(index=False):
        sl = slice(int(slide.offset), int(slide.offset + slide.n_nuclei))
        windows = slide_windows(
            x[sl], y[sl], labels[sl], codes[sl], cell_tissue[slide.slide_id], regions
        )
        mean_c = np.stack(windows["mean_c"].to_numpy())
        frac_active = np.stack(windows["frac_active"].to_numpy())
        dense = windows["dense"].to_numpy()
        tissue_ok = windows["tissue_fraction"].to_numpy() >= regions.min_tissue_fraction
        qualifies = dense & tissue_ok
        n_nuclei = windows["n_nuclei"].to_numpy()
        carcinoma = windows["frac_carcinoma"].to_numpy() >= 0.5

        positive = mean_c > 0
        considered += positive.sum(0)
        rejected += (positive & ~dense[:, None]).sum(0)
        rejected_tissue += (positive & (dense & ~tissue_ok)[:, None]).sum(0)

        for j in range(num_concepts):
            picks = {}
            for kind, mask in (
                ("carcinoma", qualifies & carcinoma),
                ("benign", qualifies & ~carcinoma),
            ):
                if mask.any():
                    picks[kind] = np.flatnonzero(mask)[mean_c[mask, j].argmax()]
            if dense.any():
                order = np.lexsort((-n_nuclei[dense], mean_c[dense, j]))
                picks["absent"] = np.flatnonzero(dense)[order[0]]
            fallback = ~qualifies & positive[:, j]
            if fallback.any():
                picks["fallback"] = np.flatnonzero(fallback)[
                    n_nuclei[fallback].argmax()
                ]
            for kind, w in picks.items():
                rows.append(
                    {
                        "concept": j,
                        "kind": kind,
                        "slide_id": slide.slide_id,
                        "x0_um": windows.at[w, "x0_um"],
                        "y0_um": windows.at[w, "y0_um"],
                        "n_nuclei": windows.at[w, "n_nuclei"],
                        "tissue_fraction": windows.at[w, "tissue_fraction"],
                        "sparse": not dense[w],
                        "low_tissue": not tissue_ok[w],
                        "frac_carcinoma": windows.at[w, "frac_carcinoma"],
                        "mean_c": mean_c[w, j],
                        "frac_active": frac_active[w, j],
                    }
                )
    rejections = pd.DataFrame(
        {
            "concept": np.arange(num_concepts),
            "windows_with_activation": considered,
            "rejected_below_min_nuclei": rejected,
            "rejected_below_min_tissue": rejected_tissue,
        }
    )
    return pd.DataFrame(rows), rejections


def select_regions(candidates: pd.DataFrame, num_high: int) -> pd.DataFrame:
    """Regions shown per concept.

    High regions from different slides, alternating carcinoma- and benign-majority by
    availability, then one absent region from another slide.
    """
    selected = []
    for _, group in candidates.groupby("concept"):
        high = {
            kind: group[group["kind"] == kind].sort_values("mean_c", ascending=False)
            for kind in ("carcinoma", "benign")
        }
        used: set[str] = set()
        picks = []
        order = ["carcinoma", "benign"] + ["any"] * num_high
        for kind in order:
            if len(picks) == num_high:
                break
            pool = (
                pd.concat(high.values()).sort_values("mean_c", ascending=False)
                if kind == "any"
                else high[kind]
            )
            pool = pool[~pool["slide_id"].isin(used) & (pool["mean_c"] > 0)]
            if len(pool):
                picks.append(pool.iloc[0].copy())
                used.add(pool.iloc[0]["slide_id"])
        # fewer than num_high qualifying regions: the densest non-qualifying windows,
        # marked by the rule they fail (`sparse`, `low_tissue`)
        fallback = group[(group["kind"] == "fallback") & ~group["slide_id"].isin(used)]
        fallback = fallback.sort_values(["n_nuclei", "mean_c"], ascending=False)
        for _, pick in fallback.iterrows():
            if len(picks) == num_high:
                break
            if pick["slide_id"] not in used:
                picks.append(pick.copy())
                used.add(pick["slide_id"])
        for rank, pick in enumerate(picks):
            pick["region"] = f"high_{rank + 1}"
            pick["fallback"] = pick["kind"] == "fallback"
            selected.append(pick)

        absent = group[(group["kind"] == "absent") & ~group["slide_id"].isin(used)]
        absent = absent.sort_values(["mean_c", "n_nuclei"], ascending=[True, False])
        if len(absent):
            pick = absent.iloc[0].copy()
            pick["region"] = "absent" if pick["mean_c"] == 0 else "lowest"
            pick["fallback"] = False
            selected.append(pick)
    return pd.DataFrame(selected).reset_index(drop=True)


class SlideReader(PredictionDataset):
    """Reads WSI regions and nucleus polygons of the test slides."""

    def __init__(self, slides: pd.DataFrame) -> None:
        super().__init__(metadata=slides, embedding_mode="efd")
        self.slides = slides.set_index("slide_id")
        self.nuclei = lru_cache(maxsize=8)(
            lambda slide_id: self.get_nuclei(
                self.slides.loc[slide_id, "slide_nuclei_path"]
            )
        )

    def read_um(
        self, slide_id: str, x0_um: float, y0_um: float, size_um: float
    ) -> Image.Image:
        slide = self.slides.loc[slide_id]
        box = Box(
            round(x0_um / slide.mpp_x),
            round(y0_um / slide.mpp_y),
            round((x0_um + size_um) / slide.mpp_x),
            round((y0_um + size_um) / slide.mpp_y),
        )
        with OpenSlide(slide.slide_path) as wsi:
            slide_size = SlideSize(*wsi.dimensions)
            region = self.read_region(wsi, self.clip_box(box, slide_size))
        return Image.fromarray(self.extract_patch(region, box, slide_size))

    def polygons_um(
        self, slide_id: str, positions: np.ndarray, ids: np.ndarray
    ) -> list:
        """Polygons (in µm) of the nuclei at `positions` within the slide's index."""
        slide = self.slides.loc[slide_id]
        nuclei = self.nuclei(slide_id).iloc[positions]
        assert (nuclei["id"].to_numpy() == ids).all(), f"Index mismatch: {slide_id}"
        mpp = np.array([slide.mpp_x, slide.mpp_y])
        return [
            np.asarray(p, np.float64).reshape(-1, 2) * mpp for p in nuclei["polygon"]
        ]


def fallback_tag(region: tuple) -> str:
    if not region.fallback:
        return ""
    reasons = [name for name in ("sparse", "low_tissue") if getattr(region, name)]
    return f" (FALLBACK: {', '.join(reasons)})"


def draw_region(
    ax_he: plt.Axes,
    ax_map: plt.Axes,
    region: tuple,
    reader: SlideReader,
    test: Activations,
    slides: pd.DataFrame,
    c: np.ndarray,
    norm: Normalize,
    cmap: matplotlib.colors.Colormap,
    regions: DictConfig,
) -> None:
    """H&E of a region (`ax_he`) and its nucleus polygons filled by `c` (`ax_map`).

    `slides` is indexed by slide_id; `c` holds one concept's codes of all test nuclei.
    """
    size = regions.size_um
    slide = slides.loc[region.slide_id]
    sl = slice(int(slide.offset), int(slide.offset + slide.n_nuclei))
    x = test.index["centroid_x_um"].to_numpy()[sl]
    y = test.index["centroid_y_um"].to_numpy()[sl]
    margin = regions.polygon_margin_um
    inside = np.flatnonzero(
        (x >= region.x0_um - margin)
        & (x < region.x0_um + size + margin)
        & (y >= region.y0_um - margin)
        & (y < region.y0_um + size + margin)
    )
    rows = inside + sl.start
    polygons = reader.polygons_um(
        region.slide_id, inside, test.index["nucleus_id"].to_numpy()[rows]
    )
    polygons = [p - [region.x0_um, region.y0_um] for p in polygons]
    colors = cmap(norm(c[rows]))
    colors[c[rows] <= 0] = to_rgba(regions.inactive_color)

    he = reader.read_um(region.slide_id, region.x0_um, region.y0_um, size)
    ax_he.imshow(he, extent=(0, size, size, 0))
    ax_map.add_collection(
        PolyCollection(polygons, facecolors=colors, edgecolors="none")
    )
    for ax in (ax_he, ax_map):
        ax.set_xlim(0, size)
        ax.set_ylim(size, 0)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])


def render_concept_map(
    j: int,
    regions: pd.DataFrame,
    reader: SlideReader,
    test: Activations,
    slides: pd.DataFrame,
    codes: np.ndarray,
    summary_row: pd.Series,
    vmax: float,
    config: DictConfig,
) -> plt.Figure:
    size = config.regions.size_um
    cmap = plt.get_cmap(config.regions.cmap)
    norm = Normalize(0.0, vmax)
    slides = slides.set_index("slide_id")

    fig, axes = plt.subplots(
        len(regions),
        2,
        figsize=(10, 5 * len(regions) + 0.8),
        squeeze=False,
        layout="constrained",
    )
    for row, region in enumerate(regions.itertuples(index=False)):
        draw_region(
            axes[row, 0],
            axes[row, 1],
            region,
            reader,
            test,
            slides,
            codes[:, j],
            norm,
            cmap,
            config.regions,
        )
        axes[row, 0].set_title(
            f"{region.region}{fallback_tag(region)} · {region.slide_id[:8]} · "
            f"carcinoma {region.frac_carcinoma:.0%} · n={region.n_nuclei} · "
            f"tissue {region.tissue_fraction:.0%}",
            fontsize=9,
            loc="left",
        )
        axes[row, 1].set_title(
            f"mean c={region.mean_c:.3g} · active {region.frac_active:.0%}",
            fontsize=9,
            loc="left",
        )

    fig.suptitle(
        f"Concept {j} · wᵀu={summary_row.logit_weight:+.3f} · "
        f"share |φ|={summary_row.share_abs_phi:.1%} · "
        f"freq={summary_row.activation_frequency:.1%} · {size:g} µm regions",
        fontsize=11,
    )
    fig.colorbar(
        ScalarMappable(norm, cmap),
        ax=axes,
        orientation="horizontal",
        fraction=0.02,
        label=f"c_{j} (inactive = gray; clipped at q{config.regions.vmax_quantile})",
    )
    return fig


def nucleus_crop(
    nucleus: tuple,
    reader: SlideReader,
    offsets: pd.Series,
    size_um: float,
    tile_px: int,
) -> Image.Image:
    """H&E crop of `size_um` around a nucleus (a `top_nuclei` row), outlined in green."""
    half = size_um / 2
    x0, y0 = nucleus.centroid_x_um - half, nucleus.centroid_y_um - half
    image = reader.read_um(nucleus.slide_id, x0, y0, size_um)
    position = nucleus.row - int(offsets[nucleus.slide_id])
    (polygon,) = reader.polygons_um(
        nucleus.slide_id, np.array([position]), np.array([nucleus.nucleus_id])
    )
    scale = np.array(image.size) / size_um
    outline = ((polygon - [x0, y0]) * scale).ravel().tolist()
    ImageDraw.Draw(image).line(outline + outline[:2], fill=(0, 255, 0), width=2)
    return image.resize((tile_px, tile_px), Image.BILINEAR)


def render_gallery(
    top: pd.DataFrame,
    reader: SlideReader,
    test: Activations,
    slides: pd.DataFrame,
    gallery: DictConfig,
) -> Image.Image:
    tile, caption = gallery.tile_px, 14
    cols = int(np.ceil(np.sqrt(gallery.num)))
    rows = int(np.ceil(len(top) / cols)) or 1
    canvas = Image.new("RGB", (cols * tile, rows * (tile + caption)), "white")
    draw = ImageDraw.Draw(canvas)
    offsets = slides.set_index("slide_id")["offset"]
    for n, nucleus in enumerate(top.itertuples(index=False)):
        image = nucleus_crop(nucleus, reader, offsets, gallery.size_um, tile)
        x, y = (n % cols) * tile, (n // cols) * (tile + caption)
        canvas.paste(image, (x, y))
        label = "carc." if nucleus.label == 1 else "benign"
        draw.text((x + 2, y + tile), f"c={nucleus.c:.2f} {label}", fill="black")
    return canvas


README = """# Concept material: k={k}, seed {seed}

Sparse autoencoder `{model}` (λ = {l1:g}), selected for k={k} as the seed with the
lowest held-out validation RE. MLflow run: `{run_id}` (child of `{parent}`).
All statistics are over all {n:,} test nuclei, unweighted.

| File | Content |
|---|---|
| `concept_summary.csv` | One row per concept, see below |
| `codes_test.npy` | Concept codes c (n x k, float32), rows aligned with `activations/test/index.parquet` |
| `contrib_test.npy` | Exact contributions φ_j = (wᵀu_j) c_j (n x k, float32), same rows; b + Σ_j φ_j = f_A with b = {baseline:.6f} |
| `boundary_distance_um_test.npy` | Distance of every test nucleus to the tissue-mask boundary (µm, float32; 0 outside tissue), same rows |
| `top_nuclei.parquet` | Per concept, the {top_n} test nuclei with the highest c_j, at most {max_per_slide} per slide |
| `regions.parquet` | The regions shown in the concept maps, with their statistics |
| `region_rejections.csv` | Per concept: windows with positive mean activation, and how many the nuclei rule and the tissue rule rejected |
| `concept_maps/concept_<j>.png` | H&E (left) and nucleus polygons filled by c_j (right) for each region |
| `gallery/concept_<j>.png` | The top {gallery_num} nuclei (as in `top_nuclei.parquet`) with {gallery_um:g} µm context, nucleus outlined |
| `regions_v1.parquet`, `region_rejections_v1.csv`, `concept_maps_v1/` | Supplementary: the first version, chosen without the tissue rule (step 1 below) |

## `concept_summary.csv`

| Column | Meaning |
|---|---|
| `activation_frequency` | Share of nuclei with c_j > 0 |
| `mean_c_carcinoma`, `mean_c_benign` | Mean c_j over carcinoma / benign nuclei (dense annotation labels) |
| `logit_weight`, `logit_weight_sign` | wᵀu_j and its sign (the direction of the concept's effect on the logit) |
| `cosine_u_w` | Cosine between the decoder direction u_j and the head weights w |
| `mean_abs_phi`, `share_abs_phi` | Mean abs(φ_j) and its share of Σ_j mean abs(φ_j) (importance) |
| `carcinoma_share_active` | Share of carcinoma nuclei among the nuclei with c_j > 0 |
| `spearman_c_f` | Spearman correlation of c_j with the logit f, over all nuclei |
| `boundary_activation_share` | Share of Σ c_j on nuclei within {boundary_um:g} µm of the tissue-mask boundary (for reference: {near_boundary_share_nuclei:.1%} of all nuclei, {near_boundary_share_activation_all:.1%} of the activation of all concepts) |
| `color_scale_max` | Upper end of the concept's color scale in the maps |
| `best_match_cosine_seed<s>`, `matched_concept_seed<s>` | Hungarian-matched decoder cosine to the other seeds' SAEs; `unstable` = cosine < {stability_threshold} for any other seed |

## Tissue mask

`{tissue_masks}/<slide_id>.tiff` (0/255, aligned with level 0 of the WSI), read at
the coarsest pyramid level with ≤ {mask_um_per_px:g} µm/px. The boundary distance is the
Euclidean distance transform of the mask (the image border counts as boundary).

## How the regions were chosen

1. Every test slide is covered by {size:g} x {size:g} µm windows on a {stride:g} µm grid
   (nucleus centroids). For high regions, a window qualifies if it contains at least
   {min_nuclei} nuclei **and** at least {min_tissue:.0%} of its area is tissue in the tissue mask.
2. For each concept and slide, the candidates are the qualifying window with the
   highest mean c_j among carcinoma-majority windows (≥ 50 % carcinoma nuclei by
   the dense annotation labels), the same among benign-majority windows, and the
   window with at least {min_nuclei} nuclei with the lowest mean c_j (most nuclei on ties).
3. High regions ({num_high}, from different slides, mean c_j > 0): first the best
   carcinoma-majority candidate, then the best benign-majority candidate from
   another slide, then the best remaining candidates of either kind. If fewer than
   {num_high} qualify, the densest non-qualifying windows with positive mean activation
   (from other slides) fill the rest; they are marked `fallback` in `regions.parquet`,
   with `sparse` (< {min_nuclei} nuclei) and/or `low_tissue` (tissue < {min_tissue:.0%}), and
   "(FALLBACK: …)" in the maps.
4. Absent region: from a slide not used above, the window with at least {min_nuclei} nuclei
   and the lowest mean c_j ("absent" if c_j = 0 for all its nuclei, otherwise
   "lowest"); the tissue rule is not applied.
5. The color scale of a concept is shared by its regions: 0 to the {quantile} quantile of
   the positive c_j over all test nuclei; inactive nuclei (c_j = 0) are light gray.
"""


@with_cli_args(["+xai=concept_material"])
@hydra.main(config_path="../configs", config_name="xai", version_base=None)
@autolog
def main(config: DictConfig, logger: MLFlowLogger) -> None:
    tag_parent_run(logger.run_id, config.parent_run_id)
    for key, value in setup_precision(config.matmul_precision).items():
        logger.experiment.set_tag(logger.run_id, key, value)

    head = LinearHead.from_npz(str(Path(config.activations_dir) / "head.npz"))
    test = load_activations(config.activations_dir, "test")
    slides = pd.read_parquet(config.slides)
    slides = slides[slides["split"] == "test"].sort_values("offset")
    model, meta = select_model(config)
    logger.log_hyperparams(
        {"selected_seed": meta["seed"], "l1_coeff": meta["l1_coeff"]}
    )

    codes = encode(model.to(config.device), test.h).numpy()
    U = model.decoder_weight.detach().double().cpu().numpy()

    out_dir = Path(config.output_dir) / f"k{meta['k']}_seed{meta['seed']}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # tissue mask: boundary distance of every test nucleus, cell tissue fractions
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
    boundary_distance = np.concatenate([distance for distance, _ in tissue])
    cell_tissue = {
        slide_id: cells
        for slide_id, (_, cells) in zip(slides["slide_id"], tissue, strict=True)
    }
    np.save(out_dir / "boundary_distance_um_test.npy", boundary_distance)
    near_boundary = boundary_distance <= config.tissue.boundary_um
    f = np.concatenate([head(test.z(chunk)) for chunk in test.chunks()])
    baseline_near = {
        "near_boundary_share_nuclei": float(near_boundary.mean()),
        "near_boundary_share_activation_all": float(
            codes[near_boundary].sum(dtype=np.float64) / codes.sum(dtype=np.float64)
        ),
    }
    logger.log_metrics(baseline_near)
    print(baseline_near)

    summary = concept_summary(codes, test.labels, U, head, f, near_boundary)
    vmax = np.array(
        [
            np.quantile(c[c > 0], config.regions.vmax_quantile)
            if (c > 0).any()
            else 1.0
            for c in codes.T
        ]
    )
    summary["color_scale_max"] = vmax
    summary = pd.concat([summary, seed_stability(config, meta, U)], axis=1)
    summary.to_csv(out_dir / "concept_summary.csv", index=False)
    logger.log_metrics({"unstable_concepts": int(summary["unstable"].sum())})
    np.save(out_dir / "codes_test.npy", codes.astype(np.float32))
    np.save(
        out_dir / "contrib_test.npy",
        (codes * concept_logit_weights(U, head)).astype(np.float32),
    )

    top = top_nuclei(codes, test, config.top_n, config.max_per_slide)
    top.to_parquet(out_dir / "top_nuclei.parquet", index=False)

    candidates, rejections = region_candidates(
        test, slides, codes, cell_tissue, config.regions
    )
    rejections.to_csv(out_dir / "region_rejections.csv", index=False)
    regions = select_regions(candidates, config.regions.num_high)
    regions.to_parquet(out_dir / "regions.parquet", index=False)
    for row in rejections.itertuples(index=False):
        num_fallback = int(regions[regions["concept"] == row.concept]["fallback"].sum())
        print(
            f"concept {row.concept}: of {row.windows_with_activation:,} windows with "
            f"activation, {row.rejected_below_min_nuclei:,} rejected "
            f"(< {config.regions.min_nuclei} nuclei), {row.rejected_below_min_tissue:,} "
            f"rejected (tissue < {config.regions.min_tissue_fraction}); "
            f"fallback regions shown: {num_fallback}"
        )
        logger.log_metrics(
            {
                f"concept_{row.concept}/windows_rejected": row.rejected_below_min_nuclei,
                f"concept_{row.concept}/windows_rejected_tissue": (
                    row.rejected_below_min_tissue
                ),
                f"concept_{row.concept}/windows_with_activation": row.windows_with_activation,
                f"concept_{row.concept}/fallback_regions": num_fallback,
            }
        )

    reader = SlideReader(slides)
    (out_dir / "concept_maps").mkdir(exist_ok=True)
    (out_dir / "gallery").mkdir(exist_ok=True)
    for j in range(codes.shape[1]):
        concept_regions = regions[regions["concept"] == j]
        if len(concept_regions):
            fig = render_concept_map(
                j,
                concept_regions,
                reader,
                test,
                slides,
                codes,
                summary.iloc[j],
                vmax[j],
                config,
            )
            fig.savefig(out_dir / "concept_maps" / f"concept_{j:02d}.png", dpi=100)
            plt.close(fig)

        concept_top = top[top["concept"] == j].nsmallest(config.gallery.num, "rank")
        if len(concept_top):
            render_gallery(concept_top, reader, test, slides, config.gallery).save(
                out_dir / "gallery" / f"concept_{j:02d}.png"
            )

    (out_dir / "README.md").write_text(
        README.format(
            k=meta["k"],
            seed=meta["seed"],
            model=meta["path"],
            l1=meta["l1_coeff"],
            run_id=logger.run_id,
            parent=config.parent_run_id,
            n=len(test),
            baseline=float(head(model.decoder_bias.detach().double().cpu().numpy())),
            top_n=config.top_n,
            max_per_slide=config.max_per_slide,
            min_nuclei=config.regions.min_nuclei,
            gallery_num=config.gallery.num,
            gallery_um=config.gallery.size_um,
            size=config.regions.size_um,
            stride=config.regions.stride_um,
            num_high=config.regions.num_high,
            quantile=config.regions.vmax_quantile,
            stability_threshold=config.stability_threshold,
            min_tissue=config.regions.min_tissue_fraction,
            boundary_um=config.tissue.boundary_um,
            tissue_masks=config.tissue.masks,
            mask_um_per_px=config.tissue.target_um_per_px,
            **baseline_near,
        )
    )
    # the per-nucleus arrays are large, they stay in `output_dir`
    for path in out_dir.iterdir():
        if path.suffix != ".npy":
            if path.is_dir():
                logger.log_artifacts(str(path), f"{out_dir.name}/{path.name}")
            else:
                logger.log_artifact(str(path), out_dir.name)


if __name__ == "__main__":
    main()
