from collections.abc import Iterable

import pandas as pd
from hydra.utils import instantiate
from lightning import LightningDataModule
from mlflow.artifacts import download_artifacts
from nuclei_graph.data.supervision import build_supervision
from omegaconf import DictConfig
from ratiopath.model_selection import train_test_split
from torch.utils.data import DataLoader

from nuclei_graph.data.utils import (
    collate_fn,
    collate_fn_predict,
    min_count_filter,
)
from nuclei_graph.nuclei_graph_typing import (
    Batch,
    PredictBatch,
)


BASE_METADATA_COLS = [
    "slide_id",
    "is_carcinoma",
    "slide_nuclei_path",
    "slide_path",
    "mpp_x",
    "mpp_y",
]

TRAIN_METADATA_COLS = [*BASE_METADATA_COLS, "patient_id", "nuclei_count"]


class DataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        eval_strategy: DictConfig,
        train_strategy: DictConfig | None = None,
        split_size: float = 0.1,
        num_workers: int = 0,
        sampler: DictConfig | None = None,
        **data_params: DictConfig,
    ) -> None:
        """Lightning DataModule for nuclei point cloud datasets with weak supervision.

        Args:
            batch_size: Batch size for training.
            num_workers: Number of workers for data loading. Defaults to 0.
            eval_strategy: A DictConfig defining the type of supervision to use for evaluation (validation or test/predict).
            train_strategy: A DictConfig defining the type of supervision to use for positive slides during training.
            split_size: Proportion of the training data to use for validation. Defaults to 0.1.
            sampler: Sampler configuration for training data loader. Defaults to None.
            **data_params: Additional parameters expected to contain keys:
                - dataset: DictConfig for instantiation of a Torch Dataset.
                - mlflow_uris: DictConfig with MLflow keys "supervision" and "metadata" containing URIs for respective artifacts.
        """
        super().__init__()
        self.batch_size = batch_size
        self.train_strategy = instantiate(train_strategy)
        self.eval_strategy = instantiate(eval_strategy)
        self.split_size = split_size
        self.num_workers = num_workers
        self.sampler_partial = sampler
        self.dataset_cfg = data_params["dataset"]
        self.uris_cfg = data_params["mlflow_uris"]
        self.positivity: dict[str, float] = {}

    def _load_df(self, uri: str, cols: list[str] | None = None) -> pd.DataFrame:
        path = download_artifacts(uri)
        return pd.read_parquet(path, columns=cols)

    def _get_labels_df(
        self, uri: str | None, slide_ids: set[str] | None = None
    ) -> pd.DataFrame | None:
        if uri is None:
            return None
        df = self._load_df(uri)
        return df[df["slide_id"].isin(slide_ids)] if slide_ids is not None else df

    def _get_sup(self, strategy, slide_df, slide_ids=None):
        slide_label_map = {
            str(k): int(v)
            for k, v in slide_df.set_index("slide_id")["is_carcinoma"].items()
        }
        return build_supervision(
            sup_strategy=strategy,
            label_map=slide_label_map,
            df_annot=self._get_labels_df(strategy.annot_uri, slide_ids),
            df_cam=self._get_labels_df(strategy.cam_uri, slide_ids),
            df_pred=self._get_labels_df(strategy.pred_uri, slide_ids),
        )

    def setup(self, stage: str) -> None:
        mode = "train" if stage in {"fit", "validate"} else stage
        slides_uri = self.uris_cfg.metadata[mode]

        match stage:
            case "fit" | "validate":
                assert self.train_strategy is not None
                slides_df = self._load_df(slides_uri, cols=TRAIN_METADATA_COLS)
                slides_df = slides_df.sort_values(by="slide_id").reset_index(drop=True)

                train_df, validation_df = train_test_split(
                    slides_df,
                    test_size=self.split_size,
                    random_state=42,
                    stratify=slides_df["is_carcinoma"],
                    groups=slides_df["patient_id"],
                )

                train_df = train_df.reset_index(drop=True)
                train_df = min_count_filter(train_df, self.dataset_cfg.crop_size)

                ids = set(train_df["slide_id"])
                sup_train = self._get_sup(self.train_strategy, train_df, ids)
                self.positivity = {
                    slide_id: slide_sup.nuclei_supervision.get_positivity()
                    for slide_id, slide_sup in sup_train.supervision_map.items()
                }

                self.train_dataset = instantiate(
                    self.dataset_cfg,
                    slides=train_df,
                    supervision=sup_train,
                )

                validation_df = validation_df.reset_index(drop=True)
                ids = set(validation_df["slide_id"])
                self.validation_dataset = instantiate(
                    self.dataset_cfg,
                    slides=validation_df,
                    supervision=self._get_sup(self.eval_strategy, validation_df, ids),
                    full_slide=True,
                )

            case "test" | "predict":
                slides_df = self._load_df(slides_uri, cols=BASE_METADATA_COLS)
                dataset = instantiate(
                    self.dataset_cfg,
                    slides=slides_df,
                    supervision=self._get_sup(self.eval_strategy, slides_df),
                    full_slide=True,
                    predict=(stage == "predict"),
                )
                if stage == "test":
                    self.test_dataset = dataset
                else:
                    self.predict_dataset = dataset

    def train_dataloader(self) -> Iterable[Batch]:
        sampler = (
            instantiate(self.sampler_partial, slides_positivity=self.positivity)(
                dataset=self.train_dataset
            )
            if self.sampler_partial is not None
            else None
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=sampler is None,
            collate_fn=collate_fn,
            drop_last=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> Iterable[Batch]:
        return DataLoader(
            self.validation_dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=collate_fn,
        )

    def test_dataloader(self) -> Iterable[Batch]:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=collate_fn,
        )

    def predict_dataloader(self) -> Iterable[PredictBatch]:
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=collate_fn_predict,
        )
