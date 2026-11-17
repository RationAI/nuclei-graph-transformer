"""Multi-provider slide tables, slide-label-only supervision and weight initialization."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from nuclei_graph.data.datamodules.base import split_train_val
from nuclei_graph.data.datamodules.crop import CropDataModule
from nuclei_graph.data.samplers import AutoWeightedRandomSampler
from nuclei_graph.data.supervision import (
    SlideLabelNucleiSupervision,
    SupervisionStrategy,
)
from nuclei_graph.meta_archs.crop import CropModelMetaArch


def slides(prefix: str, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gleason = rng.choice(["0+0", "3+3", "3+4", "4+4"], size=n)
    return pd.DataFrame(
        {
            "slide_id": [f"{prefix}{i:03d}" for i in range(n)],
            "gleason_score": gleason,
            "is_carcinoma": gleason != "0+0",
            "nuclei_count": rng.integers(100, 1000, size=n),
        }
    )


@pytest.fixture
def tables(tmp_path: Path) -> dict[str, str]:
    paths = {}
    for name, (prefix, n, seed) in {
        "a_train": ("a", 60, 0),
        "a_test": ("at", 20, 1),
        "b_train": ("b", 80, 2),
        "b_test": ("bt", 30, 3),
    }.items():
        paths[name] = str(tmp_path / f"{name}.parquet")
        slides(prefix, n, seed).to_parquet(paths[name])
    return paths


def datamodule(metadata: dict, supervision: dict | None = None) -> CropDataModule:
    return CropDataModule(
        batch_size=2,
        num_workers=0,
        eval_num_workers=0,
        metadata=OmegaConf.create(metadata),
        dataset=OmegaConf.create({}),
        block_size=128,
        k=4,
        supervision=OmegaConf.create(supervision) if supervision else None,
        split_stratify_col="gleason_score",
        split_group_col=None,
        split_size=0.1,
    )


def test_single_provider_unchanged(tables: dict[str, str]) -> None:
    dm = datamodule({"train": tables["a_train"], "test": tables["a_test"]})
    train, valid = dm.load_train_val_dfs()
    expected = split_train_val(
        pd.read_parquet(tables["a_train"]), 0.1, "gleason_score", None
    )
    pd.testing.assert_frame_equal(train, expected[0])
    pd.testing.assert_frame_equal(valid, expected[1])
    assert "provider" not in dm.load_df(tables["a_test"])


def test_two_providers_split_each_on_its_own(tables: dict[str, str]) -> None:
    dm = datamodule(
        {
            "train": {"a": tables["a_train"], "b": tables["b_train"]},
            "test": {"a": tables["a_test"]},
        }
    )
    train, valid = dm.load_train_val_dfs()
    for provider in ["a", "b"]:
        alone = split_train_val(
            pd.read_parquet(tables[f"{provider}_train"]), 0.1, "gleason_score", None
        )
        for got, expected in [(train, alone[0]), (valid, alone[1])]:
            part = got[got.provider == provider].drop(columns="provider")
            pd.testing.assert_frame_equal(part.reset_index(drop=True), expected)
    assert len(train) == 54 + 72 and len(valid) == 6 + 8
    assert not set(train["slide_id"]) & set(valid["slide_id"])
    # the other providers' test tables are not read
    assert not set(train["slide_id"]) & set(pd.read_parquet(tables["b_test"]).slide_id)

    test = dm.load_df(dm.metadata_uris_cfg["test"], cols=["slide_id"])
    assert list(test.columns) == ["slide_id", "provider"]
    assert (test.provider == "a").all()


def test_per_provider_supervision(tables: dict[str, str], tmp_path: Path) -> None:
    # dense nucleus labels of provider a only; provider b has slide labels only
    a = pd.read_parquet(tables["a_train"])
    labels = pd.DataFrame(
        [
            {"slide_id": s, "id": f"n{j}", "annot_label": float(j % 2)}
            for s in a.loc[a.is_carcinoma, "slide_id"]
            for j in range(3)
        ]
    )
    labels.to_parquet(tmp_path / "labels.parquet")
    annotation = {
        "_target_": "nuclei_graph.data.supervision.SupervisionStrategy",
        "mode": "annotation",
        "annot_dir": str(tmp_path / "labels.parquet"),
    }
    slide_label = {
        "_target_": "nuclei_graph.data.supervision.SupervisionStrategy",
        "mode": "slide_label",
    }
    dm = datamodule(
        {"train": {"a": tables["a_train"], "b": tables["b_train"]}},
        {
            "train_strategy": slide_label,
            "eval_strategy": {"a": annotation, "b": slide_label},
        },
    )
    assert isinstance(dm.train_strategy, SupervisionStrategy)
    assert isinstance(dm.eval_strategy, dict)
    train, valid = dm.load_train_val_dfs()

    sup = dm.supervise(valid, dm.eval_strategy)
    for row in valid.itertuples():
        nuclei = sup.supervision_map[row.slide_id].nuclei_supervision
        if row.provider == "b":
            assert isinstance(nuclei, SlideLabelNucleiSupervision)
            assert not nuclei.get_sup_mask(5).any()
        elif row.is_carcinoma:
            assert nuclei.get_targets(3).tolist() == [0.0, 1.0, 0.0]
            assert nuclei.get_sup_mask(3).all()

    # training: no label file is read, and the sampler follows the slide labels
    train_sup = dm.supervise(train, dm.train_strategy)
    assert train_sup.positivity_map == {
        r.slide_id: float(r.is_carcinoma) for r in train.itertuples()
    }

    class Dataset:
        metadata = train

        def __len__(self) -> int:
            return len(train)

    weights = torch.tensor(
        AutoWeightedRandomSampler(
            Dataset(),  # type: ignore[arg-type]
            train_sup.positivity_map,
            positivity_thr=0.0,
            replacement=True,
            pos_slide_ratio=0.5,
        ).weights
    )
    pos = torch.from_numpy(train["is_carcinoma"].to_numpy())
    assert torch.isclose(weights[pos].sum(), torch.tensor(0.5, dtype=weights.dtype))


def test_slide_label_supervision() -> None:
    strategy = SupervisionStrategy(mode="slide_label")
    pos = strategy.create(True)
    neg = strategy.create(False)
    assert not pos.get_sup_mask(4).any() and not neg.get_sup_mask(4).any()
    assert pos.get_targets(4).tolist() == [1.0] * 4
    assert neg.get_targets(4).tolist() == [0.0] * 4
    assert pos.get_positivity() == 1.0 and neg.get_positivity() == 0.0
    assert pos.get_pos_seeds(4) == [] and neg.get_neg_seeds(4) == []
    with pytest.raises(NotImplementedError):
        pos.get_pos_count()


class TinyNet(nn.Module):
    def __init__(self, extra: bool = False) -> None:
        super().__init__()
        self.proj = nn.Linear(3, 4)
        self.norm = nn.BatchNorm1d(4)
        if extra:
            self.head = nn.Linear(4, 1)


def test_init_ckpt_loads_weights_only(tmp_path: Path) -> None:
    torch.manual_seed(0)
    source = CropModelMetaArch(lr=1e-3, warmup_epochs=1, net=TinyNet())
    source.net.norm.running_mean.fill_(3.0)
    path = tmp_path / "source.ckpt"
    torch.save(
        {
            "epoch": 7,
            "state_dict": source.state_dict(),
            "current_model_state": {},  # as saved under EMAWeightAveraging
            "optimizer_states": [{"not": "used"}],
        },
        path,
    )

    torch.manual_seed(1)
    target = CropModelMetaArch(
        lr=1e-3, warmup_epochs=1, net=TinyNet(extra=True), init_ckpt=str(path)
    )
    for name, value in source.net.state_dict().items():
        assert torch.equal(target.net.state_dict()[name], value), name
    report = target.init_report
    assert report is not None
    assert report["init_ckpt_epoch"] == 7 and report["init_ckpt_ema"]
    assert report["init_loaded_keys"] == len(source.net.state_dict())
    assert report["init_missing_keys"] == ["head.bias", "head.weight"]
    assert report["init_unexpected_keys"] == []

    # a shape mismatch is an error, not a silent partial load
    class WideNet(TinyNet):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(3, 5)
            self.norm = nn.BatchNorm1d(5)

    with pytest.raises(RuntimeError):
        CropModelMetaArch(lr=1e-3, warmup_epochs=1, net=WideNet(), init_ckpt=str(path))

    # without init_ckpt nothing is loaded
    assert (
        CropModelMetaArch(lr=1e-3, warmup_epochs=1, net=TinyNet()).init_report is None
    )
