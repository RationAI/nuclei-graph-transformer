"""Study small_foci: focus table, seeding (A1), losses (A2, A3) and self-evidence head (B1).

The defaults must keep the current behaviour.
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from nuclei_graph.data.datamodules.collator import GraphCollator
from nuclei_graph.data.datamodules.crop import CropDataModule
from nuclei_graph.data.datasets.crop.nuclei_classification import (
    NucleiClassificationDataset,
)
from nuclei_graph.data.foci import FOCI_COLUMNS, FociTable, nucleus_keys, slide_key
from nuclei_graph.data.supervision import (
    AnnotationNucleiSupervision,
    DatasetSupervision,
    SlideSupervision,
)
from nuclei_graph.meta_archs.nuclei import (
    NucleiModelMetaArch,
    crop_balanced_bce,
    focus_weighted_bce,
)
from nuclei_graph.modeling.models.transformer import Transformer


# --------------------------------------------------------------------------
# Synthetic slide: 600 nuclei, a large focus (300 carcinoma nuclei) and a small one (12)
# --------------------------------------------------------------------------

N, BIG, SMALL = 600, 300, 12


def make_slide(tmp_path: Path) -> dict:
    rng = np.random.default_rng(0)
    side = int(np.ceil(np.sqrt(N)))
    grid = np.stack(np.meshgrid(np.arange(side), np.arange(side)), -1).reshape(-1, 2)[
        :N
    ]
    centroids = grid * 20.0 + rng.uniform(-3, 3, size=(N, 2))
    t = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    polygons = [
        np.stack([c[0] + a * np.cos(t), c[1] + b * np.sin(t)], 1).ravel()
        for c, a, b in zip(
            centroids, rng.uniform(3, 6, N), rng.uniform(2, 4, N), strict=True
        )
    ]
    ids = [rng.bytes(32).hex() for _ in range(N)]
    float_list = pa.list_(pa.float32())  # as the nuclei tables: list<float>
    pq.write_table(
        pa.table(
            {
                "id": pa.array(ids),
                "polygon": pa.array(
                    [q.astype(np.float32) for q in polygons], float_list
                ),
                "centroid": pa.array(
                    [c.astype(np.float32) for c in centroids], float_list
                ),
            }
        ),
        path := tmp_path / "nuclei.parquet",
    )
    nuclei = pd.read_parquet(path)
    nuclei = nuclei.sort_values("id", ignore_index=True)  # get_nuclei order
    # spatial foci: the BIG left-most nuclei, and the SMALL nuclei nearest the far corner
    xy = np.stack(nuclei["centroid"].tolist())
    focus = np.zeros(N, np.int32)
    focus[np.argsort(xy[:, 0], kind="stable")[:BIG]] = 1
    corner = np.linalg.norm(xy - xy.max(0), axis=1)
    focus[np.argsort(corner, kind="stable")[:SMALL]] = 2
    assert (focus == 1).sum() == BIG and (focus == 2).sum() == SMALL
    labels = (focus > 0).astype(np.float32)
    meta = pd.DataFrame(
        {
            "slide_id": ["s0"],
            "slide_nuclei_path": [str(path)],
            "slide_path": [""],
            "is_carcinoma": [True],
            "mpp_x": [0.5],
            "mpp_y": [0.5],
        }
    )
    sup = DatasetSupervision(
        {
            "s0": SlideSupervision(
                1, AnnotationNucleiSupervision(True, torch.from_numpy(labels))
            )
        }
    )
    car = np.flatnonzero(focus > 0)
    sizes = np.where(focus[car] == 1, BIG, SMALL).astype(np.int32)
    table = pd.DataFrame(
        {
            "slide_key": np.full(len(car), slide_key("s0"), np.uint64),
            "slide_nuclei": np.full(len(car), N, np.int32),
            "index": car.astype(np.int32),
            "key": nucleus_keys(nuclei["id"].to_numpy()[car]),
            "focus": focus[car],
            "focus_nuclei": sizes,
        }
    )[list(FOCI_COLUMNS)]
    foci_path = tmp_path / "foci.parquet"
    table.to_parquet(foci_path)
    return {
        "meta": meta,
        "sup": sup,
        "foci": str(foci_path),
        "ids": nuclei["id"].to_numpy(),
        "focus": focus,
        "labels": labels,
    }


def dataset(slide: dict, **kwargs) -> NucleiClassificationDataset:
    return NucleiClassificationDataset(
        metadata=slide["meta"],
        embedding_mode="efd",
        supervision=slide["sup"],
        crop_size_min=40,
        crop_size_max=80,
        efd_order=4,
        **kwargs,
    )


def seed_all(s: int) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


@pytest.fixture
def slide(tmp_path: Path) -> dict:
    return make_slide(tmp_path)


# --------------------------------------------------------------------------
# Focus table
# --------------------------------------------------------------------------


def test_foci_table_sizes_and_id_check(slide: dict) -> None:
    table = FociTable(slide["foci"])
    sizes = table.focus_sizes("s0", N, slide["ids"])
    expected = np.where(
        slide["focus"] == 1, BIG, np.where(slide["focus"] == 2, SMALL, 0)
    )
    assert np.array_equal(sizes, expected)
    assert table.get("other") is None
    assert np.array_equal(table.focus_sizes("other", 5), np.zeros(5))
    wrong = slide["ids"].copy()
    c, b = np.flatnonzero(slide["focus"] > 0)[0], np.flatnonzero(slide["focus"] == 0)[0]
    wrong[c], wrong[b] = (
        wrong[b],
        wrong[c],
    )  # a carcinoma nucleus swapped with a benign one
    with pytest.raises(AssertionError, match="carcinoma nuclei differ"):
        FociTable(slide["foci"]).get("s0", wrong)
    wrong = slide["ids"].copy()
    c0, c1 = np.flatnonzero(slide["focus"] > 0)[:2]
    wrong[c0], wrong[c1] = wrong[c1], wrong[c0]  # two carcinoma nuclei reordered
    with pytest.raises(AssertionError, match="carcinoma nuclei differ"):
        FociTable(slide["foci"]).get("s0", wrong)
    with pytest.raises(AssertionError, match="nuclei"):
        FociTable(slide["foci"]).get("s0", slide["ids"][:-1])


def test_focus_seed_uniform_over_foci(slide: dict) -> None:
    table = FociTable(slide["foci"])
    pos = np.flatnonzero(slide["labels"] == 1).tolist()
    random.seed(0)
    seeds = [table.focus_seed("s0", pos, slide["ids"])[0] for _ in range(4000)]
    in_small = np.mean(slide["focus"][seeds] == 2)
    assert abs(in_small - 0.5) < 0.03  # uniform over nuclei: 12 / 312 = 0.04
    assert set(seeds) <= set(pos)
    # only foci holding a supervised positive are drawn; no foci -> uniform over the seeds
    only_big = [p for p in pos if slide["focus"][p] == 1]
    assert all(
        slide["focus"][table.focus_seed("s0", only_big)[0]] == 1 for _ in range(200)
    )
    assert table.focus_seed("other", [3, 4]) in ([3], [4])
    assert table.focus_seed("s0", []) == []


# --------------------------------------------------------------------------
# Dataset and collator
# --------------------------------------------------------------------------


def test_default_dataset_draws_unchanged_by_foci(slide: dict) -> None:
    """Loading the focus table (seeding by nucleus) adds labels but no random draw."""
    plain, with_foci = dataset(slide), dataset(slide, foci=slide["foci"])
    assert plain.foci is None and plain.pos_seed_mode == "nucleus"
    for s in range(20):
        seed_all(s)
        a = plain[0]
        seed_all(s)
        b = with_foci[0]
        for key in ("features", "pos", "sup_mask", "seq_len"):
            assert torch.equal(a[key], b[key]), key
        assert torch.equal(a["labels"]["nuclei"], b["labels"]["nuclei"])
        assert "focus_nuclei" not in a["labels"]
        sizes = b["labels"]["focus_nuclei"]
        assert torch.equal(sizes > 0, b["labels"]["nuclei"] == 1)
        assert set(sizes.unique().tolist()) <= {0.0, float(BIG), float(SMALL)}


def test_focus_seeding_reaches_the_small_focus(slide: dict) -> None:
    """Crops grown from focus seeds contain the small focus far more often."""
    counts = {}
    for mode in ("nucleus", "focus"):
        ds = dataset(slide, foci=slide["foci"], pos_seed_mode=mode)
        seed_all(0)
        hits = 0
        for _ in range(300):
            sizes = ds[0]["labels"]["focus_nuclei"]
            hits += int((sizes == SMALL).any())
        counts[mode] = hits
    assert counts["focus"] > 2 * counts["nucleus"] + 20, counts


def test_focus_seed_mode_needs_foci(slide: dict) -> None:
    with pytest.raises(AssertionError, match="focus table"):
        dataset(slide, pos_seed_mode="focus")
    with pytest.raises(AssertionError, match="pos_seed_mode"):
        dataset(slide, foci=slide["foci"], pos_seed_mode="slide")


def test_collator_packs_focus_sizes_with_labels(slide: dict) -> None:
    ds = dataset(slide, foci=slide["foci"])
    seed_all(1)
    samples = [ds[0] for _ in range(3)]
    batch = GraphCollator(block_size=16, k=4, predict=False, context="knn")(samples)
    n = int(batch["seq_lens"].sum())
    sizes, labels = batch["labels"]["focus_nuclei"], batch["labels"]["nuclei"]
    assert sizes.shape == labels.shape
    assert torch.equal(sizes[:n] > 0, labels[:n] == 1)
    assert (sizes[n:] == 0).all()
    plain = GraphCollator(block_size=16, k=4, predict=False, context="knn")(
        [dataset(slide)[0]]
    )
    assert "focus_nuclei" not in plain["labels"]


def test_eval_datasets_drop_the_focus_options() -> None:
    dm = CropDataModule.__new__(CropDataModule)
    dm.dataset_cfg = OmegaConf.create({"foci": "x", "pos_seed_mode": "focus"})
    assert dm.eval_dataset_overrides() == {"foci": None, "pos_seed_mode": "nucleus"}
    dm.dataset_cfg = OmegaConf.create({})
    assert dm.eval_dataset_overrides() == {}


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------


def bce(z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return nn.functional.binary_cross_entropy_with_logits(z, y, reduction="none")


def test_focus_weighted_bce() -> None:
    z = torch.tensor([0.3, -1.0, 2.0, 0.5, -0.2])
    y = torch.tensor([1.0, 1.0, 0.0, 1.0, 0.0])
    sizes = torch.tensor([4.0, 100.0, 0.0, 25.0, 0.0])
    w = sizes[y == 1] ** -0.5
    weights = torch.ones(5)
    weights[y == 1] = w / w.mean()
    assert torch.isclose(weights[y == 1].mean(), torch.tensor(1.0))
    expected = (weights * bce(z, y)).mean()
    assert torch.allclose(focus_weighted_bce(z, y, sizes, 0.5), expected)
    # equal focus sizes (or power 0): the mean BCE
    same = torch.where(y == 1, 7.0, 0.0)
    assert torch.allclose(focus_weighted_bce(z, y, same, 0.5), bce(z, y).mean())
    assert torch.allclose(focus_weighted_bce(z, y, sizes, 0.0), bce(z, y).mean())
    # no positives: the mean BCE; a positive without a focus size is an error
    assert torch.allclose(
        focus_weighted_bce(z, torch.zeros(5), sizes, 0.5), bce(z, torch.zeros(5)).mean()
    )
    with pytest.raises(AssertionError):
        focus_weighted_bce(z, y, torch.zeros(5), 0.5)


def test_crop_balanced_bce() -> None:
    z = torch.tensor([0.3, -1.0, 2.0, 0.5, -0.2, 1.5, -2.0])
    y = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    crop = torch.tensor([0, 0, 0, 1, 1, 2, 2])  # crop 2: negatives only
    b = bce(z, y)
    c0 = b[0] + (b[1] + b[2]) / 2
    c1 = b[3] + b[4]
    c2 = (b[5] + b[6]) / 2
    expected = (c0 + c1 + c2) / 3
    assert torch.allclose(crop_balanced_bce(z, y, crop, 3), expected)
    # a crop without supervised nuclei is left out
    assert torch.allclose(crop_balanced_bce(z, y, crop, 4), expected)


class Net(nn.Module):
    def forward(self, **kwargs) -> dict:
        n = int(kwargs["seq_lens"].sum())
        x = kwargs["x"][:n]
        return {"nuclei": x[:, :1], "graph": x[:1, 0], "attn_weights": x[:, 0]}


def batch_for_loss() -> dict:
    torch.manual_seed(0)
    n, pad = 11, 5
    labels = (torch.rand(n + pad) > 0.5).float()
    labels[n:] = 0
    sup = torch.ones(n + pad, dtype=torch.bool)
    sup[[2, 7]] = False
    return {
        "features": torch.randn(n + pad, 3),
        "pos": torch.zeros(n + pad, 2),
        "block_mask": None,
        "seq_lens": torch.tensor([4, 7], dtype=torch.int32),
        "global_neighbor_idx": torch.zeros(n + pad, 1, dtype=torch.long),
        "sup_mask": sup,
        "labels": {
            "nuclei": labels,
            "graph": None,
            "focus_nuclei": torch.where(labels == 1, 9.0, 0.0),
        },
    }


@pytest.mark.parametrize("loss", ["bce", "focus_weighted", "crop_balanced"])
def test_meta_arch_training_loss(loss: str) -> None:
    arch = NucleiModelMetaArch(lr=1e-4, warmup_epochs=1, net=Net(), train_loss=loss)
    arch.log = lambda *a, **k: None  # type: ignore[method-assign]
    batch = batch_for_loss()
    n = int(batch["seq_lens"].sum())
    sup = batch["sup_mask"][:n]
    z = batch["features"][:n, 0][sup]
    y = batch["labels"]["nuclei"][:n][sup]
    got = arch.training_step(batch)
    if loss in (
        "bce",
        "focus_weighted",
    ):  # equal focus sizes: focus_weighted = mean BCE
        assert torch.allclose(got, nn.BCEWithLogitsLoss()(z, y))
    else:
        crop = torch.repeat_interleave(torch.arange(2), torch.tensor([4, 7]))[sup]
        assert torch.allclose(got, crop_balanced_bce(z, y, crop, 2))


def test_meta_arch_rejects_unknown_loss() -> None:
    with pytest.raises(AssertionError):
        NucleiModelMetaArch(lr=1e-4, warmup_epochs=1, net=Net(), train_loss="focal")


# --------------------------------------------------------------------------
# Self-evidence head (B1)
# --------------------------------------------------------------------------


def net_config(**extra) -> dict:
    return OmegaConf.create(
        {
            "efd_order": 4,
            "node_features": 19,
            "norm_dim": 17,
            "dim": 32,
            "hidden_dim": 64,
            "num_heads": 4,
            "num_layers": 2,
            "num_classes": 1,
            "drop_path_rate": 0.0,
            "embedding_mode": "efd",
            "context": "none",  # self-only (needs the standard variant): B1 is independent of it
            "use_rope": True,
            "attention_variant": "standard",
            "pooling": "attention",
            **extra,
        }
    )


def test_self_head_default_keeps_the_model() -> None:
    torch.manual_seed(0)
    without_key = Transformer(net_config())
    torch.manual_seed(0)
    default = Transformer(net_config(self_head=False))
    assert default.self_head is None
    a, b = without_key.state_dict(), default.state_dict()
    assert a.keys() == b.keys()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_self_head_adds_a_self_logit() -> None:
    torch.manual_seed(0)
    base = Transformer(net_config())
    torch.manual_seed(0)
    model = Transformer(net_config(self_head=True))
    sd_base, sd = base.state_dict(), model.state_dict()
    assert set(sd) - set(sd_base) == {"self_head.weight", "self_head.bias"}
    assert all(torch.equal(sd_base[k], sd[k]) for k in sd_base)  # created last

    model.eval()
    n, total = 10, 16
    x = torch.randn(total, 19)
    pos = torch.randn(total, 2)
    seq_lens = torch.tensor([6, 4])
    out = model(x=x, pos=pos, block_mask=None, seq_lens=seq_lens)
    assert out["nuclei"].shape == (n, 1)
    assert torch.allclose(out["nuclei"], out["nuclei_context"] + out["nuclei_self"])
    h0 = model.prepare_features(x, pos, None, n, seq_lens)
    assert torch.allclose(out["nuclei_self"], model.self_head(h0[:n]))
    # the context logit is the base model's logit (same parameters)
    base.eval()
    ref = base(x=x, pos=pos, block_mask=None, seq_lens=seq_lens)
    assert torch.allclose(ref["nuclei"], out["nuclei_context"], atol=1e-6)
    assert "nuclei_self" not in ref

    model.train()
    model(x=x, pos=pos, block_mask=None, seq_lens=seq_lens)["nuclei"].sum().backward()
    assert model.self_head.weight.grad is not None
    assert model.class_head.weight.grad is not None
