"""Script for computing the total positive and negative nuclei counts for the weighted loss."""

from pathlib import Path

import hydra
import pandas as pd
import torch
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig
from pandas import DataFrame


def load_nuclei(nuclei_path: str) -> dict[str, Path]:
    slides_nuclei = list(Path(nuclei_path).rglob("*.pt"))
    return {slide_nuclei.stem: slide_nuclei for slide_nuclei in slides_nuclei}


def load_annotations(
    slides: DataFrame, annot_masks_path: str | None
) -> dict[str, Path]:
    annotations: dict[str, Path] = {}
    if annot_masks_path is None:
        return annotations

    annot_masks = Path(annot_masks_path)
    for annot_mask in annot_masks.rglob("*.pt"):
        if annot_mask.stem in slides.slide_id.values:
            annotations[annot_mask.stem] = annot_mask
    return annotations


def compute_slides_positivity(
    slides: DataFrame, nuclei: dict[str, Path], annotations: dict[str, Path]
) -> None:
    """Computes total positive and negative nuclei counts across slides.

    For positive slides, only labeled positive nuclei are counted.
    For negative slides, all the nuclei are counted as negative.
    """
    total_pos_nuclei = 0
    total_neg_nuclei = 0

    for idx in range(len(slides)):
        slide_id = slides.iloc[idx].slide_id
        mask_path = annotations.get(slide_id)
        nuclei_path = nuclei[slide_id]

        nuclei_data = torch.load(nuclei_path, weights_only=False, mmap=True)
        labels = nuclei_data["y"]

        if mask_path is not None:
            annot_mask = torch.load(mask_path, weights_only=False, mmap=True).bool()
            labels = labels[annot_mask]
            total_pos_nuclei += labels.sum().item()
        else:
            total_neg_nuclei += nuclei_data["y"].shape[0]

    print(f"Total positive nuclei in dataset: {total_pos_nuclei}")
    print(f"Total negative nuclei in dataset: {total_neg_nuclei}")
    print(f"Overall neg/pos ratio: {total_neg_nuclei / total_pos_nuclei:.4f}")


@hydra.main(config_path="../../configs", config_name="preprocessing", version_base=None)
def main(config: DictConfig) -> None:
    metadata_path = download_artifacts(config.efd_transform.train_uri)
    nuclei_path = download_artifacts(config.efd_mean_std.train_uri)
    annot_masks_path = (
        download_artifacts(config.nuclei_masks.cam_masks_uri)
        if config.nuclei_masks.cam_masks_uri is not None
        else None
    )

    slides = pd.read_parquet(metadata_path)
    nuclei = load_nuclei(nuclei_path)
    annotations = load_annotations(slides, annot_masks_path)

    compute_slides_positivity(slides, nuclei, annotations)


if __name__ == "__main__":
    main()
