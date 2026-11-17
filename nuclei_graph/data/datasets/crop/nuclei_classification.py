from typing import Any

import numpy as np
import torch

from nuclei_graph.data.datasets.crop.base import BaseCropDataset
from nuclei_graph.data.foci import FociTable
from nuclei_graph.nuclei_graph_typing import Sample, Targets


POS_SEED_MODES = ("nucleus", "focus")


class NucleiClassificationDataset(BaseCropDataset):
    """Crops (training) or whole slides (evaluation) with nucleus labels.

    Args:
        foci: URI or path of the focus table (`nuclei_graph.data.foci`). With it, each sample's
            labels also hold `focus_nuclei`, the size (carcinoma nuclei) of each nucleus's focus
            (0 outside carcinoma), for the focus-size loss weight.
        pos_seed_mode: How the seed of a positive crop is drawn: "nucleus" uniform over the
            slide's positive nuclei; "focus" a focus uniformly, then a positive nucleus within
            it (needs `foci`).
    """

    def __init__(
        self,
        *args: Any,
        foci: str | None = None,
        pos_seed_mode: str = "nucleus",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert pos_seed_mode in POS_SEED_MODES, (
            f"Invalid pos_seed_mode: {pos_seed_mode}"
        )
        assert pos_seed_mode == "nucleus" or foci is not None, (
            "pos_seed_mode='focus' needs the focus table (foci)"
        )
        self.pos_seed_mode = pos_seed_mode
        self.foci = FociTable(foci) if foci is not None else None

    def __getitem__(self, idx: int) -> Sample:
        slide = self.metadata.iloc[idx]
        nuclei = self.get_nuclei(slide.slide_nuclei_path, self.nuclei_columns)
        centroids = self.get_centroids(nuclei, slide.mpp_x, slide.mpp_y)

        crop_indices = np.arange(len(nuclei), dtype=int)
        ids = nuclei["id"].to_numpy()
        nuclei_sup = self.get_nuclei_sup(slide.slide_id, ids)

        # Crop Generation
        if not self.full_slide:
            target_size = self.get_crop_size(len(nuclei))

            if not slide.is_carcinoma:
                crop_indices = self.get_crop_indices(
                    centroids, nuclei_sup.get_neg_seeds(len(nuclei)), target_size
                )
            else:
                # Target nuclei positivity distribution:
                # 50% positive, 25% negative (from pos slides), 25% negative (from neg slides).
                # Assuming sampler yields 75% positive slides, select a positive seed with proba 2/3.
                valid_seeds = (
                    self.pos_seeds(slide.slide_id, nuclei_sup, ids)
                    if torch.rand(1).item() < (2.0 / 3.0)
                    else nuclei_sup.get_neg_seeds(len(nuclei))
                )
                crop_indices = self.get_crop_indices(
                    centroids, valid_seeds, target_size
                )
        assert crop_indices is not None

        # Supervision
        crop_indices_t = torch.from_numpy(crop_indices).long()
        crop_nuclei_labels = nuclei_sup.get_targets(len(nuclei))[crop_indices_t]
        labels = Targets(nuclei=crop_nuclei_labels, graph=None)
        if self.foci is not None:
            sizes = self.foci.focus_sizes(slide.slide_id, len(nuclei), ids)
            labels["focus_nuclei"] = torch.from_numpy(sizes)[crop_indices_t]

        # Geometry & Augmentations
        crop_polygons, crop_pos, crop_pos_centered = self.process_geometry(
            nuclei, crop_indices, centroids, slide
        )

        # Embeddings
        crop_geom_features, crop_bboxes = self.generate_embeddings(
            crop_polygons, crop_pos, slide, nuclei, crop_indices
        )

        return Sample(
            {
                "features": torch.as_tensor(crop_geom_features, dtype=torch.float32)
                if crop_geom_features is not None
                else None,
                "bboxes": crop_bboxes,
                "labels": labels,
                "pos": torch.as_tensor(crop_pos_centered, dtype=torch.float32),
                "sup_mask": nuclei_sup.get_sup_mask(len(nuclei))[crop_indices_t],
                "seq_len": torch.tensor(len(crop_indices), dtype=torch.int32),
                "metadata": None,
            }
        )

    def pos_seeds(self, slide_id: str, nuclei_sup: Any, ids: np.ndarray) -> list[int]:
        """Candidate seeds of a positive crop (one, drawn by focus, with `pos_seed_mode="focus"`)."""
        seeds = nuclei_sup.get_pos_seeds(len(ids))
        if self.pos_seed_mode == "nucleus":
            return seeds
        assert self.foci is not None
        return self.foci.focus_seed(slide_id, seeds, ids)
