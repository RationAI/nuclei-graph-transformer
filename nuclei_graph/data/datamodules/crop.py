from collections.abc import Iterable

import pandas as pd
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from nuclei_graph.data.datamodules.base import METADATA_COLS_EVAL, BaseDataModule
from nuclei_graph.data.datamodules.collator import GraphCollator
from nuclei_graph.nuclei_graph_typing import Batch


class CropDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.positivity: dict[str, float] = {}

    def min_count_filter(self, df: pd.DataFrame, min_count: int) -> pd.DataFrame:
        """Filters rows in the input dataframe based on a minimum count of nuclei.

        Args:
            df: Input DataFrame with columns "slide_nuclei_path" (str), "slide_id" (str), and "nuclei_count" (int).
            min_count: Minimum number of nuclei required to retain the slide.
        """
        mask_keep = df["nuclei_count"] >= min_count
        if not mask_keep.all():
            dropped_slides = df.loc[~mask_keep, ["slide_id", "nuclei_count"]].copy()
            print(
                f"[INFO] Dropped slides with < {min_count} nuclei:\n"
                f"{dropped_slides.to_string(index=False)}"
            )
        return df[mask_keep].reset_index(drop=True)

    def min_positive_count_filter(
        self,
        df: pd.DataFrame,
        min_pos_count: float,
        pos_counts: dict[str, int],
    ) -> pd.DataFrame:
        """Filters positive slides if their absolute number of positive nuclei is strictly less than `min_pos_count`.

        Args:
            df: Input DataFrame with "slide_id" and "is_carcinoma" columns.
            min_pos_count: Minimum absolute number of positive nuclei required to retain a positive slide.
            pos_counts: Dictionary mapping slide IDs to their count of confident positive nuclei.
        """
        pos_count = df["slide_id"].map(pos_counts)
        mask_keep = (~df["is_carcinoma"]) | pos_count.ge(min_pos_count)

        if not mask_keep.all():
            dropped_slides = df.loc[~mask_keep, ["slide_id", "nuclei_count"]].copy()
            dropped_slides["pos_nuclei_count"] = pos_count[~mask_keep]

            print(
                f"[INFO] Dropped slides with < {min_pos_count} positive nuclei:\n"
                f"{dropped_slides.to_string(index=False)}"
            )

        return df[mask_keep].reset_index(drop=True)

    def setup(self, stage: str) -> None:
        mode = "train" if stage in {"fit", "validate"} else stage
        slides_uri = self.metadata_uris_cfg[mode]

        match stage:
            case "fit" | "validate":
                assert self.split_size is not None
                assert self.eval_strategy is not None

                slides_df = self.load_df(slides_uri)
                train_df, validation_df = self.get_train_val_dfs(slides_df)

                if stage == "fit":
                    assert self.train_strategy is not None

                    if self.dataset_cfg.get("crop_size_min") is not None:
                        train_df = self.min_count_filter(
                            train_df, self.dataset_cfg.crop_size_min
                        )
                    train_sup = self.prepare_supervision(
                        train_df, self.train_strategy.paths, self.train_strategy
                    )
                    self.positivity = train_sup.positivity_map

                    if self.dataset_cfg.get("crop_pos_thr") is not None:
                        min_pos_count = (
                            self.dataset_cfg.crop_size_min
                            * self.dataset_cfg.crop_pos_thr
                        )
                        train_df = self.min_positive_count_filter(
                            train_df, min_pos_count, train_sup.pos_count_map
                        )
                    self.train_dataset = instantiate(
                        self.dataset_cfg,
                        metadata=train_df,
                        supervision=train_sup,
                        full_slide=True,
                    )

                validation_sup = self.prepare_supervision(
                    validation_df, self.eval_strategy.paths, self.eval_strategy
                )
                self.validation_dataset = instantiate(
                    self.dataset_cfg,
                    metadata=validation_df,
                    supervision=validation_sup,
                    augmentations=None,
                    full_slide=True,
                )
            case "test":
                slides_df = self.load_df(
                    slides_uri,
                    cols=[*METADATA_COLS_EVAL, "is_carcinoma"],
                )
                assert self.eval_strategy is not None
                sup = self.prepare_supervision(
                    slides_df, self.eval_strategy.paths, self.eval_strategy
                )
                self.test_dataset = instantiate(
                    self.dataset_cfg,
                    metadata=slides_df,
                    supervision=sup,
                    augmentations=None,
                    full_slide=True,
                )
            case "predict":
                slides_df = self.load_df(slides_uri, cols=METADATA_COLS_EVAL)
                self.predict_dataset = instantiate(
                    self.dataset_cfg, metadata=slides_df
                )

    def train_dataloader(self) -> Iterable[Batch]:
        sampler = None
        if self.sampler_cfg is not None:
            sampler_fn = instantiate(
                self.sampler_cfg, slides_positivity=self.positivity
            )
            sampler = sampler_fn(dataset=self.train_dataset)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            collate_fn=GraphCollator(
                block_size=self.block_size, k=self.k, predict=False
            ),
            drop_last=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable[Batch]:
        return DataLoader(
            self.validation_dataset,
            batch_size=1,
            num_workers=self.eval_num_workers,
            persistent_workers=self.eval_num_workers > 0,
            prefetch_factor=2 if self.eval_num_workers > 0 else None,
            collate_fn=GraphCollator(
                block_size=self.block_size, k=self.k, predict=False
            ),
        )

    def test_dataloader(self) -> Iterable[Batch]:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.eval_num_workers,
            persistent_workers=self.eval_num_workers > 0,
            prefetch_factor=2 if self.eval_num_workers > 0 else None,
            collate_fn=GraphCollator(
                block_size=self.block_size, k=self.k, predict=False
            ),
        )

    def predict_dataloader(self) -> Iterable[Batch]:
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            num_workers=self.eval_num_workers,
            persistent_workers=self.eval_num_workers > 0,
            prefetch_factor=2 if self.eval_num_workers > 0 else None,
            collate_fn=GraphCollator(
                block_size=self.block_size, k=self.k, predict=True
            ),
        )
