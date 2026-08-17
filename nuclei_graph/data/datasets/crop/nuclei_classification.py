import numpy as np
import torch

from nuclei_graph.data.datasets.crop.base import BaseCropDataset
from nuclei_graph.nuclei_graph_typing import Sample


class NucleiClassificationDataset(BaseCropDataset):
    def __getitem__(self, idx: int) -> Sample:
        slide = self.metadata.iloc[idx]
        nuclei = self.get_nuclei(slide.slide_nuclei_path)
        centroids = self.get_centroids(nuclei, slide.mpp_x, slide.mpp_y)

        crop_indices = np.arange(len(nuclei), dtype=int)
        nuclei_sup = self.get_nuclei_sup(slide.slide_id)

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
                    nuclei_sup.get_pos_seeds(len(nuclei))
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

        # Geometry & Augmentations
        crop_polygons, crop_pos, crop_pos_centered = self.process_geometry(
            nuclei, crop_indices, centroids, slide
        )

        # Embeddings
        crop_geom_features, crop_bboxes = self.generate_embeddings(
            crop_polygons, crop_pos, slide
        )

        return Sample(
            {
                "features": torch.as_tensor(crop_geom_features, dtype=torch.float32)
                if crop_geom_features is not None
                else None,
                "bboxes": crop_bboxes,
                "labels": {"nuclei": crop_nuclei_labels, "graph": None},
                "pos": torch.as_tensor(crop_pos_centered, dtype=torch.float32),
                "sup_mask": nuclei_sup.get_sup_mask(len(nuclei))[crop_indices_t],
                "seq_len": torch.tensor(len(crop_indices), dtype=torch.int32),
                "metadata": None,
            }
        )
