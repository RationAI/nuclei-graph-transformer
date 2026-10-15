import json

import numpy as np
import pandas as pd
import pytest
import torch
from lightning import LightningDataModule, Trainer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from nuclei_graph.callbacks.activations import ActivationsCallback
from nuclei_graph.data.datamodules.collator import GraphCollator
from nuclei_graph.meta_archs.nuclei import NucleiModelMetaArch
from nuclei_graph.modeling.models import Transformer
from nuclei_graph.nuclei_graph_typing import Sample
from xai.prepare_nuclei import ids_checksum


EFD_ORDER = 2
SLIDES = {"slide_a": 150, "slide_b": 90}


def make_sample(slide_id: str, n: int, rng: np.random.Generator) -> Sample:
    ids = np.array([f"{slide_id}_{i:04d}" for i in rng.permutation(n)])
    return Sample(
        {
            "features": torch.from_numpy(
                rng.normal(size=(n, 4 * EFD_ORDER + 3))
            ).float(),
            "bboxes": None,
            "labels": {"nuclei": None, "graph": None},
            "pos": torch.from_numpy(rng.uniform(0, 200, size=(n, 2))).float(),
            "sup_mask": torch.ones(n, dtype=torch.bool),
            "seq_len": torch.tensor(n, dtype=torch.int32),
            "metadata": {"slide_id": slide_id, "nuclei_ids": ids},
        }
    )


def make_model(context: str, attention_variant: str) -> NucleiModelMetaArch:
    torch.manual_seed(0)
    config = OmegaConf.create(
        {
            "efd_order": EFD_ORDER,
            "node_features": 4 * EFD_ORDER + 3,
            "norm_dim": 4 * EFD_ORDER + 1,
            "dim": 32,
            "hidden_dim": 64,
            "num_heads": 2,
            "num_layers": 2,
            "num_classes": 1,
            "drop_path_rate": 0.0,
            "embedding_mode": "efd",
            "context": context,
            "attention_variant": attention_variant,
            "use_rope": True,
        }
    )
    return NucleiModelMetaArch(lr=1e-4, warmup_epochs=1, net=Transformer(config))


@pytest.mark.parametrize(
    ("context", "attention_variant", "save_layers"),
    [("none", "standard", False), ("none", "standard", True), ("knn", "vrope", True)],
)
def test_hook_reproduces_logits(
    tmp_path, context: str, attention_variant: str, save_layers: bool
) -> None:
    model = make_model(context, attention_variant)
    rng = np.random.default_rng(0)
    samples = [make_sample(s, n, rng) for s, n in SLIDES.items()]

    # slides table as written by xai.prepare_nuclei: index rows in id order
    slides = pd.DataFrame(
        {
            "slide_id": list(SLIDES),
            "split": ["validation", "test"],
            "n_nuclei": list(SLIDES.values()),
            "offset": [0, 0],
            "ids_crc32": [
                ids_checksum(np.sort(s["metadata"]["nuclei_ids"])) for s in samples
            ],
        }
    )
    slides.to_parquet(tmp_path / "slides.parquet")

    callback = ActivationsCallback(
        slides_path=str(tmp_path / "slides.parquet"),
        output_dir=str(tmp_path / "activations"),
        save_layers=save_layers,
    )
    loader = DataLoader(
        samples,
        batch_size=1,
        collate_fn=GraphCollator(block_size=128, k=4, predict=True, context=context),
    )
    trainer = Trainer(
        accelerator="cpu", logger=False, callbacks=[callback], enable_progress_bar=False
    )
    predictions = trainer.predict(model, loader)

    # the callback checks class_head(h) against the logits before the float16 cast
    summary = json.loads(
        (tmp_path / "activations" / "activations_summary.json").read_text()
    )
    assert summary["class_head_max_abs_diff"] < 1e-4
    assert summary["slides_missing"] == 0

    head = np.load(tmp_path / "activations" / "head.npz")
    for split, slide in (("validation", 0), ("test", 1)):
        out = tmp_path / "activations" / split
        h, logits = np.load(out / "h.npy"), np.load(out / "logits.npy")
        n = list(SLIDES.values())[slide]
        assert h.shape == (n, 32) and h.dtype == np.float16
        assert logits.dtype == np.float32

        # rows follow the index order (sorted nucleus ids) and hold the model's
        # logits of the same nuclei
        batch_ids = loader.collate_fn([samples[slide]])["metadata"]["nuclei_ids"][0]
        model_logits = predictions[slide]["nuclei"].squeeze(-1).numpy()
        expected = pd.Series(model_logits, index=batch_ids).sort_index()
        np.testing.assert_allclose(logits, expected.to_numpy(), atol=1e-6)

        # class_head of the stored (float16) h reproduces the logits up to the cast
        np.testing.assert_allclose(
            h.astype(np.float64) @ head["w"] + head["beta"], logits, atol=5e-3
        )

        if save_layers:
            layers = np.load(out / "layers.npy")
            assert layers.shape == (n, 2, 32)
            # the last layer output, normalized, is h
            last = torch.from_numpy(layers[:, -1].astype(np.float32))
            normed = model.net.final_norm(last).detach().numpy()
            np.testing.assert_allclose(normed, h, atol=1e-2)


class SlidesDataset(Dataset):
    def __init__(self, samples: list[Sample]) -> None:
        self.samples = {s["metadata"]["slide_id"]: s for s in samples}
        self.metadata = pd.DataFrame({"slide_id": list(self.samples)})

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[self.metadata["slide_id"].iloc[idx]]


class SlidesDataModule(LightningDataModule):
    def __init__(self, samples: list[Sample]) -> None:
        super().__init__()
        self.samples = samples

    def setup(self, stage: str) -> None:
        self.predict_dataset = SlidesDataset(self.samples)

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_dataset,
            batch_size=1,
            collate_fn=GraphCollator(block_size=128, k=4, predict=True, context="none"),
        )


def test_resume_skips_finished_slides(tmp_path) -> None:
    model = make_model("none", "standard")
    rng = np.random.default_rng(0)
    samples = [make_sample(s, n, rng) for s, n in SLIDES.items()]
    pd.DataFrame(
        {
            "slide_id": list(SLIDES),
            "split": ["test", "test"],
            "n_nuclei": list(SLIDES.values()),
            "offset": [0, SLIDES["slide_a"]],
            "ids_crc32": [
                ids_checksum(np.sort(s["metadata"]["nuclei_ids"])) for s in samples
            ],
        }
    ).to_parquet(tmp_path / "slides.parquet")

    def run() -> list:
        callback = ActivationsCallback(
            slides_path=str(tmp_path / "slides.parquet"),
            output_dir=str(tmp_path / "activations"),
        )
        trainer = Trainer(
            accelerator="cpu",
            logger=False,
            callbacks=[callback],
            enable_progress_bar=False,
        )
        return trainer.predict(model, datamodule=SlidesDataModule(samples))

    out = tmp_path / "activations" / "test"
    assert len(run()) == 2
    h_full = np.load(out / "h.npy").copy()

    # simulate an interrupted run: slide_b was never finished
    (out / "done" / "slide_b.json").unlink()
    h = np.load(out / "h.npy", mmap_mode="r+")
    h[SLIDES["slide_a"] :] = 0
    h.flush()
    del h

    assert len(run()) == 1  # only slide_b is predicted again
    np.testing.assert_array_equal(np.load(out / "h.npy"), h_full)
    summary = json.loads(
        (tmp_path / "activations" / "activations_summary.json").read_text()
    )
    assert summary["slides_written"] == 2 and summary["slides_missing"] == 0
