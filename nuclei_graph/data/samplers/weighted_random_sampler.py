from collections.abc import Sequence

import torch
from torch.utils.data import WeightedRandomSampler

from nuclei_graph.data.datasets import NucleiDataset


class AutoWeightedRandomSampler(WeightedRandomSampler):
    def __init__(
        self,
        dataset: NucleiDataset,
        slides_positivity: dict[str, float],
        positivity_thr: float,
        replacement: bool,
        pos_slide_ratio: float,
    ):
        """Weighted Random Sampler with weights based on the class distribution.

        It assigns weights to each sample in the dataset based on the inverse of the class frequency.

        Arguments:
            dataset: Torch dataset class (NucleiDataset).
            slides_positivity: Dictionary mapping slide IDs to their positivity scores.
            positivity_thr: Threshold for considering a graph to be positive in the sampler.
            replacement: If True, samples are drawn with replacement. Default is True.
            pos_slide_ratio: Ratio of positive slides to include in the sampler.
        """
        weights = self._get_weights(
            dataset, slides_positivity, positivity_thr, pos_slide_ratio
        )
        super().__init__(weights, num_samples=len(dataset), replacement=replacement)

    def _get_weights(
        self,
        dataset: NucleiDataset,
        slides_positivity: dict[str, float],
        positivity_thr: float,
        pos_slide_ratio: float,
    ) -> Sequence[float]:
        slide_ids = dataset.slides["slide_id"].values
        labels = torch.tensor(
            [1.0 if slides_positivity[id] > positivity_thr else 0.0 for id in slide_ids]
        )
        positive = int(labels.sum().item())
        negative = len(labels) - positive

        if positive == 0 or negative == 0:
            raise ValueError("Both positive and negative nuclei must exist.")

        weights = torch.zeros_like(labels)
        weights[labels == 0] = (1.0 - pos_slide_ratio) / negative
        weights[labels == 1] = pos_slide_ratio / positive
        return weights.tolist()
