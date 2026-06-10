import random

import pandas as pd
from torch.utils.data import Sampler


class StratifiedInterleavedSlideSampler(Sampler):
    def __init__(
        self,
        tiles_df: pd.DataFrame,
        target_col: str = "carcinoma",
        max_active_slides: int = 3,
        samples_per_epoch: int = 10000,
    ):
        """Tile Sampler that maintains a fixed number of active slides and stratifies slide selection by class label."""
        self.tiles_df = tiles_df
        self.max_active_slides = max_active_slides
        self.target_col = target_col
        self.samples_per_epoch = min(samples_per_epoch, len(tiles_df))

        self.master_slide_groups = {
            k: list(v) for k, v in tiles_df.groupby("slide_id").indices.items()
        }

        self.slide_labels = tiles_df.groupby("slide_id")[target_col].first()

        self.remaining_tiles = {}
        self.active_slides = []

        self._reset_global_state()

    def _reset_global_state(self):
        """Reset tile pools and reshuffle tiles within each slide."""
        self.remaining_tiles = {k: list(v) for k, v in self.master_slide_groups.items()}
        for slide_id in self.remaining_tiles:
            random.shuffle(self.remaining_tiles[slide_id])

        self.active_slides = []

    def _get_next_slide(self):
        """Select a new slide from the remaining slides using inverse-frequency class weighting."""
        available_slides = [
            s
            for s, tiles in self.remaining_tiles.items()
            if len(tiles) > 0 and s not in self.active_slides
        ]

        if not available_slides:
            return None

        labels_subset = self.slide_labels.loc[pd.Index(available_slides)]
        value_counts = labels_subset.value_counts()
        weights = 1.0 / labels_subset.map(value_counts)

        return random.choices(available_slides, weights=weights.tolist(), k=1)[0]

    def __iter__(self):
        yielded_count = 0

        while yielded_count < self.samples_per_epoch:
            while len(self.active_slides) < self.max_active_slides:
                next_slide = self._get_next_slide()
                if next_slide is not None:
                    self.active_slides.append(next_slide)
                else:
                    break

            if not self.active_slides:
                self._reset_global_state()
                continue

            current_slide = random.choice(self.active_slides)
            yield self.remaining_tiles[current_slide].pop()
            yielded_count += 1

            if not self.remaining_tiles[current_slide]:
                self.active_slides.remove(current_slide)

    def __len__(self):
        return self.samples_per_epoch
