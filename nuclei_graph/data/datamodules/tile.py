from collections.abc import Iterable
from functools import partial

from hydra.utils import instantiate
from torch.utils.data import DataLoader

from nuclei_graph.data.datamodules.base import METADATA_COLS_EVAL, BaseDataModule
from nuclei_graph.data.utils import predict_collate_fn, supervised_collate_fn
from nuclei_graph.nuclei_graph_typing import LabeledSampleBatch, UnlabeledSampleBatch


class TileDataModule(BaseDataModule):
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
                    train_sup = self.prepare_supervision(
                        train_df, self.train_strategy.paths, self.train_strategy
                    )
                    self.train_dataset = instantiate(
                        self.dataset_cfg,
                        metadata=train_df,
                        supervision=train_sup,
                    )

                validation_sup = self.prepare_supervision(
                    validation_df, self.eval_strategy.paths, self.eval_strategy
                )
                self.validation_dataset = instantiate(
                    self.dataset_cfg,
                    metadata=validation_df,
                    supervision=validation_sup,
                    carcinoma_filter=False,
                )
            case "test":
                slides_df = self.load_df(
                    slides_uri, cols=[*METADATA_COLS_EVAL, "is_carcinoma"]
                )
                assert self.eval_strategy is not None
                sup = self.prepare_supervision(
                    slides_df, self.eval_strategy.paths, self.eval_strategy
                )

                self.test_dataset = instantiate(
                    self.dataset_cfg,
                    metadata=slides_df,
                    supervision=sup,
                    carcinoma_filter=False,
                )
            case "predict":
                slides_df = self.load_df(slides_uri, cols=METADATA_COLS_EVAL)
                self.predict_dataset = instantiate(
                    self.dataset_cfg,
                    metadata=slides_df,
                    carcinoma_filter=False,
                )

    def train_dataloader(self) -> Iterable[LabeledSampleBatch]:
        sampler = None
        if self.sampler_cfg is not None:
            sampler_fn = instantiate(self.sampler_cfg)
            sampler = sampler_fn(tiles_df=self.train_dataset.tiles)

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=False,
            collate_fn=partial(
                supervised_collate_fn, block_size=self.block_size, k=self.k
            ),
            drop_last=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=True,
        )

    def val_dataloader(self) -> Iterable[LabeledSampleBatch]:
        return DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.eval_num_workers,
            persistent_workers=self.eval_num_workers > 0,
            prefetch_factor=2 if self.eval_num_workers > 0 else None,
            collate_fn=partial(
                supervised_collate_fn, block_size=self.block_size, k=self.k
            ),
            pin_memory=True,
        )

    def test_dataloader(self) -> Iterable[LabeledSampleBatch]:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.eval_num_workers,
            persistent_workers=self.eval_num_workers > 0,
            prefetch_factor=2 if self.eval_num_workers > 0 else None,
            collate_fn=partial(
                supervised_collate_fn, block_size=self.block_size, k=self.k
            ),
            pin_memory=True,
        )

    def predict_dataloader(self) -> Iterable[UnlabeledSampleBatch]:
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.eval_num_workers,
            persistent_workers=self.eval_num_workers > 0,
            prefetch_factor=2 if self.eval_num_workers > 0 else None,
            collate_fn=partial(
                predict_collate_fn, block_size=self.block_size, k=self.k
            ),
            pin_memory=True,
        )
