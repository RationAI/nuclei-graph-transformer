from collections.abc import Iterable

import pandas as pd
from hydra.utils import instantiate
from lightning import LightningDataModule
from mlflow.artifacts import download_artifacts
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader

from nuclei_graph.data.datasets.nuclei_dataset import NucleiDataset
from nuclei_graph.data.utils.collator import collate_fn, collate_fn_predict
from nuclei_graph.data.utils.compute_stats import (
    compute_median_neighbor_distance,
    compute_scale_stats,
)
from nuclei_graph.data.utils.sampler import (
    compute_slides_positivity,
    pre_crop_filter,
)
from nuclei_graph.data.utils.splitter import get_subset, train_val_split
from nuclei_graph.nuclei_graph_typing import (
    PredictInput,
    Sample,
)


BASE_METADATA_COLS = [
    "slide_id",
    "is_carcinoma",
    "slide_nuclei_path",
]

TRAIN_METADATA_COLS = [*BASE_METADATA_COLS, "patient_id"]
LABEL_COLS = {"annot_label": "label"}
REFINEMENT_COLS = {"cam_thr_mask": "refinement_mask", "cam_score": "score"}


class DataModule(LightningDataModule):
    def __init__(
        self,
        seed: int,
        batch_size: int,
        num_workers: int = 0,
        sampler: DictConfig | None = None,
        **datasets: DictConfig,
    ) -> None:
        super().__init__()
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.datasets = datasets
        self.sampler_partial = sampler
        self.positivity: dict[str, float] = {}

    def _instantiate_dataset(self, conf: DictConfig, **kwargs) -> NucleiDataset:
        conf = conf.copy()
        with open_dict(conf):
            conf.pop("uris", None)
            conf.pop("stats", None)
        return instantiate(conf, **kwargs)

    def prepare_data(self) -> None:
        uris = {
            uri
            for conf in self.datasets.values()
            if isinstance(conf, DictConfig) and conf.get("uris") is not None
            for uri in conf.uris.values()
            if uri is not None
        }
        for uri in uris:
            download_artifacts(uri)

    def _get_stats(self, conf: DictConfig, df_train: pd.DataFrame) -> dict[str, float]:
        scale_mean = conf.stats.scale_mean
        scale_std = conf.stats.scale_std
        neighbor_dist_median = conf.stats.neighbor_dist_median

        if scale_mean is None or scale_std is None:
            scale_mean, scale_std = compute_scale_stats(df_train, conf.efd_order)
        if neighbor_dist_median is None:
            neighbor_dist_median = compute_median_neighbor_distance(df_train)

        return {
            "scale_mean": scale_mean,
            "scale_std": scale_std,
            "neighbor_dist_median": neighbor_dist_median,
        }

    def setup(self, stage: str) -> None:
        mode = "train" if stage in ["fit", "validate"] else stage
        conf = self.datasets[mode]

        df_labels = pd.read_parquet(download_artifacts(conf.uris.labels_uri))
        df_labels = df_labels.rename(columns=LABEL_COLS)
        df_refinement = None
        if conf.uris.get("refinement_uri") is not None:
            df_refinement = pd.read_parquet(
                download_artifacts(conf.uris.refinement_uri)
            )
            df_refinement = df_refinement.rename(columns=REFINEMENT_COLS)

        match stage:
            case "fit" | "validate":
                metadata = pd.read_parquet(
                    download_artifacts(conf.uris.metadata_uri),
                    columns=TRAIN_METADATA_COLS,
                )
                df_train, df_val = train_val_split(
                    metadata, keep_cols=TRAIN_METADATA_COLS, random_state=self.seed
                )
                df_train = pre_crop_filter(df_train, conf.crop_size)
                self.positivity = compute_slides_positivity(df_train, df_labels)
                stats = self._get_stats(conf, df_train)

                train_slides_ids = set(df_train["slide_id"])
                self.train = self._instantiate_dataset(
                    conf,
                    df_metadata=df_train,
                    df_labels=get_subset(train_slides_ids, df_labels),
                    df_refinement=get_subset(train_slides_ids, df_refinement),
                    **stats,
                )
                val_slides_ids = set(df_val["slide_id"])
                self.val = self._instantiate_dataset(
                    conf,
                    df_metadata=df_val,
                    df_labels=get_subset(val_slides_ids, df_labels),
                    df_refinement=get_subset(val_slides_ids, df_refinement),
                    **stats,
                    full_slide=True,
                )
            case "test":
                metadata = pd.read_parquet(
                    download_artifacts(conf.uris.metadata_uri),
                    columns=BASE_METADATA_COLS,
                )
                self.test = self._instantiate_dataset(
                    conf,
                    df_metadata=metadata,
                    df_labels=df_labels,
                    df_refinement=df_refinement,
                    **conf.stats,  # must be provided in the config
                )
            case "predict":
                metadata = pd.read_parquet(
                    download_artifacts(conf.uris.metadata_uri),
                    columns=BASE_METADATA_COLS,
                )
                self.predict = self._instantiate_dataset(
                    conf,
                    df_metadata=metadata,
                    df_labels=df_labels,
                    df_refinement=df_refinement,
                    **conf.stats,  # must be provided in the config
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
            batch_size=1,
            num_workers=0,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> Iterable[Sample]:
        return DataLoader(
            self.test,
            batch_size=1,
            num_workers=0,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self) -> Iterable[PredictInput]:
        return DataLoader(
            self.predict,
            batch_size=1,
            num_workers=0,
            collate_fn=collate_fn_predict,
        )
