import random
from collections.abc import Iterator

import pandas as pd
from torch.utils.data import Sampler


class BalancedInterleavedSlideSampler(Sampler):
    def __init__(
        self,
        tiles_df: pd.DataFrame,
        target_col: str = "carcinoma",
        max_active_slides: int = 4,
        samples_per_epoch: int = 10000,
    ):
        self.tiles_df = tiles_df
        self.samples_per_epoch = min(samples_per_epoch, len(tiles_df))

        self.pos_slots = max_active_slides // 2
        self.neg_slots = max_active_slides - self.pos_slots

        self.slide_labels = tiles_df.groupby("slide_id")[target_col].first()
        self.pos_slide_ids = self.slide_labels[self.slide_labels == 1].index.tolist()
        self.neg_slide_ids = self.slide_labels[self.slide_labels == 0].index.tolist()

        self.master_slide_groups = {
            k: list(v) for k, v in tiles_df.groupby("slide_id").indices.items()
        }

        self.remaining_tiles = {}
        self.active_pos = []
        self.active_neg = []

    def _reset_pool(self, pool_type: str) -> None:
        """Refills and shuffles only the specified class pool."""
        slide_ids = self.pos_slide_ids if pool_type == "pos" else self.neg_slide_ids
        for sid in slide_ids:
            tiles = list(self.master_slide_groups[sid])
            random.shuffle(tiles)
            self.remaining_tiles[sid] = tiles

    def _get_next_slide(self, pool_type: str) -> int | None:
        """Finds the next available slide for the specified class, refilling if necessary."""
        slide_ids = self.pos_slide_ids if pool_type == "pos" else self.neg_slide_ids
        active_list = self.active_pos if pool_type == "pos" else self.active_neg

        available = [
            s
            for s in slide_ids
            if len(self.remaining_tiles[s]) > 0 and s not in active_list
        ]

        if not available:
            self._reset_pool(pool_type)
            available = [
                s
                for s in slide_ids
                if len(self.remaining_tiles[s]) > 0 and s not in active_list
            ]

        return random.choice(available) if available else None

    def __iter__(self) -> Iterator[int]:
        self._reset_pool("pos")
        self._reset_pool("neg")
        self.active_pos = []
        self.active_neg = []

        yielded_count = 0
        yield_pos_next = True

        while yielded_count < self.samples_per_epoch:
            while len(self.active_pos) < self.pos_slots:
                nxt = self._get_next_slide("pos")
                if nxt is not None:
                    self.active_pos.append(nxt)

            while len(self.active_neg) < self.neg_slots:
                nxt = self._get_next_slide("neg")
                if nxt is not None:
                    self.active_neg.append(nxt)

            active_list = self.active_pos if yield_pos_next else self.active_neg
            current_slide = random.choice(active_list)

            yield self.remaining_tiles[current_slide].pop()
            yielded_count += 1
            yield_pos_next = not yield_pos_next

            if not self.remaining_tiles[current_slide]:
                active_list.remove(current_slide)

    def __len__(self):
        return self.samples_per_epoch
