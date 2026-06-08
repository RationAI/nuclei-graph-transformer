import random

import pandas as pd
from torch.utils.data import Sampler


class StratifiedInterleavedSlideSampler(Sampler):
    def __init__(
        self,
        tiles_df: pd.DataFrame,
        target_col: str = "carcinoma",
        max_active_slides: int = 3,
    ):
        self.tiles_df = tiles_df
        self.max_active_slides = max_active_slides
        self.target_col = target_col

        self.slide_groups = {
            k: list(v) for k, v in tiles_df.groupby("slide_id").indices.items()
        }

        slide_labels = tiles_df.groupby("slide_id")[target_col].first()

        value_counts = slide_labels.value_counts()
        slide_weights = 1.0 / slide_labels.map(value_counts)

        self.slide_ids = slide_labels.index.tolist()
        self.slide_weights = [float(w) for w in slide_weights.values]

        for slide_id in self.slide_groups:
            random.shuffle(self.slide_groups[slide_id])

    def __iter__(self):
        epoch_groups = {k: list(v) for k, v in self.slide_groups.items()}

        chosen_slide_order = random.choices(
            self.slide_ids, weights=self.slide_weights, k=len(self.slide_ids)
        )

        active_slides = []

        while len(active_slides) < self.max_active_slides and chosen_slide_order:
            active_slides.append(chosen_slide_order.pop(0))

        while active_slides:
            current_slide = random.choice(active_slides)

            yield epoch_groups[current_slide].pop()

            if not epoch_groups[current_slide]:
                active_slides.remove(current_slide)
                if chosen_slide_order:
                    active_slides.append(chosen_slide_order.pop(0))

    def __len__(self):
        return len(self.tiles_df)
