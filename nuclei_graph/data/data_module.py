from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from hydra.utils import instantiate
from lightning import LightningDataModule
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

from nuclei_graph.data.datasets.nuclei_dataset import NucleiDataset
from nuclei_graph.data.utils.collator import collate_fn, collate_fn_predict
from nuclei_graph.data.utils.sampler import (
    compute_slides_positivity,
    pre_crop_filter,
)
from nuclei_graph.data.utils.splitter import get_subset, train_val_split
from nuclei_graph.nuclei_graph_typing import (
    PredictInput,
    Sample,
)


class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int = 0,
        sampler: DictConfig | None = None,
        **datasets: DictConfig,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets = datasets
        self.sampler_partial = sampler
        self.positivity: dict[str, float] = {}

    def _instantiate_dataset(self, conf: DictConfig, **kwargs) -> NucleiDataset:
        conf = conf.copy()

        # remove URIs from the config before instantiation
        with open_dict(conf):
            uri_keys = [key for key in conf if str(key).endswith("_uri")]
            for uri_key in uri_keys:
                conf.pop(uri_key, None)

        return instantiate(conf, **kwargs)

    def setup(self, stage: str) -> None:
        mode = "train" if stage in ["fit", "validate"] else stage
        conf = self.datasets[mode]

        df_labels = pd.read_parquet(download_artifacts(conf.annots_uri))
        df_labels = df_labels.rename(columns={"annot_label": "label"})
        if conf.get("cam_refinement_uri") is not None:
            df_refinement = pd.read_parquet(download_artifacts(conf.cam_refinement_uri))
            df_refinement = df_refinement.rename(
                columns={"cam_thr_mask": "refinement_mask", "cam_score": "score"}
            )
        else:
            df_refinement = None

        keep_cols = ["slide_id", "is_carcinoma", "slide_nuclei_path"]
        match stage:
            case "fit" | "validate":
                keep_cols.append("patient_id")
                metadata = pd.read_parquet(
                    Path(download_artifacts(conf.metadata_uri)), columns=keep_cols
                )
                df_train, df_val = train_val_split(metadata, keep_cols=keep_cols)
                df_train = pre_crop_filter(df_train, conf.crop_size)
                self.positivity = compute_slides_positivity(
                    df_train, df_labels, df_refinement
                )
                self.train = self._instantiate_dataset(
                    conf,
                    df_metadata=df_train,
                    df_labels=get_subset(set(df_train["slide_id"]), df_labels),
                    df_refinement=get_subset(set(df_train["slide_id"]), df_refinement),
                )
                self.val = self._instantiate_dataset(
                    conf,
                    df_metadata=df_val,
                    df_labels=get_subset(set(df_val["slide_id"]), df_labels),
                    df_refinement=get_subset(set(df_val["slide_id"]), df_refinement),
                )
            case "test":
                metadata = pd.read_parquet(
                    Path(download_artifacts(conf.metadata_uri)), columns=keep_cols
                )
                self.test = self._instantiate_dataset(
                    conf,
                    df_metadata=metadata,
                    df_labels=df_labels,
                    df_refinement=df_refinement,
                )
            case "predict":
                keep_cols.append("slide_path")
                metadata = pd.read_parquet(
                    Path(download_artifacts(conf.metadata_uri)), columns=keep_cols
                )
                self.predict = self._instantiate_dataset(
                    conf,
                    df_metadata=metadata,
                    df_labels=df_labels,
                    df_refinement=df_refinement,
                )

    def train_dataloader(self) -> Iterable[Sample]:
        sampler = (
            instantiate(self.sampler_partial, slides_positivity=self.positivity)(
                dataset=self.train
            )
            if self.sampler_partial is not None
            else None
        )
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable[Sample]:
        return DataLoader(
            self.val,
            batch_size=1,  # process full graphs
            num_workers=0,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> Iterable[Sample]:
        return DataLoader(
            self.test,
            batch_size=1,  # process full graphs
            num_workers=0,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self) -> Iterable[PredictInput]:
        return DataLoader(
            self.predict,
            batch_size=1,  # process full graphs
            num_workers=0,
            collate_fn=collate_fn_predict,
        )
