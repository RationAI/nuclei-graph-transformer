"""Estimate the physical (um) size of the neighborhood a nucleus sees during crop-level
training, for k = crop_size_min (512) and k = crop_size_max (8192).

For a sample of query nuclei per slide, this reports the distance to their k-th nearest
neighbor -- i.e. the radius of the smallest ball around that nucleus containing exactly k
other nuclei -- in micrometers, averaged across queries and slides.

Note this is a simpler, more directly interpretable proxy for "how big is a training crop
physically", not an exact reproduction of the training-time sampling: the actual crop-growing
algorithm (`BaseCropDataset.get_crop_indices`/`find_component`) grows a connected component on
a Delaunay graph biased by a cost function mixing edge distance and distance-from-seed (see
`alpha` in the nuclei_level training configs), rather than taking a pure k-NN ball. A Delaunay-
biased component of size k typically spans a *larger* radius than the pure k-NN ball for the
same k, since the graph walk can wander into locally sparser regions -- so treat the numbers
here as a lower bound on the true training crop extent.

Metadata sources (see configs/base.yaml, configs/data/sources/{prostate_cancer_mmci_tl,panda}.yaml):
    mlflow_project_sc = mlflow-artifacts:/97 (sensitive cloud)
    mlflow_project_pc = mlflow-artifacts:/37 (public cloud)

Usage:
    python -m scripts.exploration.nuclei_knn_crop_radius
    python -m scripts.exploration.nuclei_knn_crop_radius --dataset mmci
    python -m scripts.exploration.nuclei_knn_crop_radius --dataset radboud karolinska --max-slides 50
"""

import argparse

import numpy as np
import pandas as pd
from mlflow.artifacts import download_artifacts
from scipy.spatial import cKDTree


K_VALUES = [512, 8192]  # crop_size_min / crop_size_max, see nuclei_level training configs

# preprocessing/metadata_mapping/{prostate_cancer_mmci_tl,panda}.py -- train split mappings
DATASET_METADATA_URIS = {
    "mmci": (
        "mlflow-artifacts:/97/28a05b3cbc2a434eae6221f103d56020/"
        "artifacts/tile_level_annotations/slides_mapping.parquet"
    ),
    "radboud": (
        "mlflow-artifacts:/37/5b9c6f096026425fa2b04c4e74b7a6f3/"
        "artifacts/panda/slides_mapping_train.parquet"
    ),
    "karolinska": (
        "mlflow-artifacts:/37/53cf24a94202486ca063607d5a588b34/"
        "artifacts/panda/slides_mapping_train.parquet"
    ),
}


def get_centroids_um(nuclei: pd.DataFrame, mpp_x: float, mpp_y: float) -> np.ndarray:
    """Nuclei centroids in micrometers (mirrors `BaseCropDataset.get_centroids`)."""
    mpps = np.array([mpp_x, mpp_y], dtype=np.float64)
    return np.stack(nuclei["centroid"].tolist()) * mpps


def knn_radii(
    centroids: np.ndarray,
    k_values: list[int],
    rng: np.random.Generator,
    max_queries: int,
) -> dict[int, np.ndarray]:
    """Distance (um) from each of a random sample of query nuclei to their k-th nearest neighbor."""
    n = len(centroids)
    tree = cKDTree(centroids)

    n_queries = min(max_queries, n)
    query_idx = rng.choice(n, size=n_queries, replace=False)
    query_points = centroids[query_idx]

    radii: dict[int, np.ndarray] = {}
    for k in k_values:
        k_eff = min(k, n - 1)  # can't have more neighbors than other nuclei exist
        dists, _ = tree.query(query_points, k=k_eff + 1)  # +1: self (distance 0) is included
        radii[k] = dists[:, -1]
    return radii


def process_dataset(
    name: str,
    uri: str,
    k_values: list[int],
    max_slides: int | None,
    max_queries: int,
    rng: np.random.Generator,
) -> None:
    print(f"\n=== {name} ===")
    slides_df = pd.read_parquet(
        download_artifacts(uri), columns=["slide_id", "slide_nuclei_path", "mpp_x", "mpp_y"]
    )

    if max_slides is not None and len(slides_df) > max_slides:
        sample_idx = rng.choice(len(slides_df), size=max_slides, replace=False)
        slides_df = slides_df.iloc[sample_idx].reset_index(drop=True)

    per_k_radii: dict[int, list[np.ndarray]] = {k: [] for k in k_values}
    for row in slides_df.itertuples(index=False):
        nuclei = pd.read_parquet(row.slide_nuclei_path, columns=["centroid"])
        if len(nuclei) < 2:
            continue

        centroids = get_centroids_um(nuclei, row.mpp_x, row.mpp_y)
        radii = knn_radii(centroids, k_values, rng, max_queries)
        for k, r in radii.items():
            per_k_radii[k].append(r)
        print(f"  {row.slide_id}: {len(nuclei)} nuclei")

    for k in k_values:
        all_r = np.concatenate(per_k_radii[k])
        print(
            f"  k={k:>5}: mean radius = {all_r.mean():.1f} um "
            f"(median {np.median(all_r):.1f}, std {all_r.std():.1f}, n_queries={len(all_r)})"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=sorted(DATASET_METADATA_URIS),
        default=sorted(DATASET_METADATA_URIS),
    )
    parser.add_argument(
        "--max-slides",
        type=int,
        default=20,
        help="Cap on slides sampled per dataset. Pass 0 to use all slides.",
    )
    parser.add_argument(
        "--max-queries-per-slide",
        type=int,
        default=200,
        help="Random nuclei sampled per slide as k-NN query points.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    max_slides = None if args.max_slides == 0 else args.max_slides
    rng = np.random.default_rng(args.seed)
    for name in args.dataset:
        process_dataset(
            name,
            DATASET_METADATA_URIS[name],
            K_VALUES,
            max_slides,
            args.max_queries_per_slide,
            rng,
        )


if __name__ == "__main__":
    main()
