from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mlflow.artifacts import download_artifacts
from omegaconf import OmegaConf
from ratiopath.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC

from nuclei_graph.configuration import Config
from nuclei_graph.data.datasets.nuclei_dataset import NucleiDataset
from nuclei_graph.data.supervision import (
    SupervisionStrategy,
    build_supervision,
)
from nuclei_graph.data.utils import supervised_collate_fn
from nuclei_graph.modeling.models.transformer import Transformer
from nuclei_graph.nuclei_wsl_meta_arch import NucleiWSLMetaArch


ANNOTS_PATH = (
    "/mnt/projects/nuclei_based_wsi_analysis/nuclei_supervision/MMCI-TL-Data/annotation"
)
MODEL_CKPT = download_artifacts(
    "mlflow-artifacts:/97/f2ea4646e1574d1c9f93c1bbda88ccde/artifacts/checkpoints/epoch=307-step=6776/checkpoint.ckpt"
)
METADATA_PATH = download_artifacts(
    "mlflow-artifacts:/97/28a05b3cbc2a434eae6221f103d56020/artifacts/tile_level_annotations/slides_mapping.parquet"
)
EFD_ORDER = 10
NEIGHBORHOOD_SIZE = 64

MODEL_CONFIG = Config(
    ffn=OmegaConf.create({}),
    self_attn=OmegaConf.create({}),
    efd_order=EFD_ORDER,
    node_features=43,
    norm_dim=41,
    dim=256,
    hidden_dim=1024,
    num_heads=8,
    num_layers=8,
    num_classes=1,
    drop_path_rate=0.15,
)

net = Transformer(MODEL_CONFIG)
model = NucleiWSLMetaArch.load_from_checkpoint(
    MODEL_CKPT, lr=0.001, warmup_epochs=0, net=net
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --- 2. INPUT PROJECTION WEIGHT ANALYSIS ---

weights = net.input_proj.weight.detach().cpu().numpy()
feature_importance = np.abs(weights).mean(axis=0)

labels = [f"EFD_{i}" for i in range(EFD_ORDER * 4)] + [
    "log_scales",
    "cos_angles",
    "sin_angles",
]

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(labels, feature_importance, color="skyblue", edgecolor="black")
ax.set_title("Average Absolute Weight Magnitude in Input Projection Layer")
ax.set_ylabel("Mean Absolute Weight")
plt.xticks(rotation=90)
plt.tight_layout()

fig.savefig("feature_importance_weights.png", dpi=1200)
plt.close(fig)
print("Saved feature_importance_weights.png")


# --- 3. PERMUTATION FEATURE IMPORTANCE ---


@torch.no_grad()
def compute_permutation_importance(
    model: NucleiWSLMetaArch, dataloader, device: torch.device
):
    print("\nStarting Permutation Feature Importance...")
    baseline_auroc = BinaryAUROC().to(device)

    def to_device(b):
        b_out = {}
        for k, v in b.items():
            if isinstance(v, torch.Tensor):
                b_out[k] = v.to(device)
            elif isinstance(v, dict):
                b_out[k] = {
                    sk: (sv.to(device) if hasattr(sv, "to") else sv)
                    for sk, sv in v.items()
                }
            elif hasattr(v, "to"):  # For BlockMask objects
                b_out[k] = v.to(device)
            else:
                b_out[k] = v
        return b_out

    # 1. Compute Baseline AUROC
    for batch in dataloader:
        batch = to_device(batch)

        # Correctly mask the targets!
        targets_sup = batch["labels"]["nuclei"][batch["sup_mask"]]
        if targets_sup.numel() == 0:
            continue

        logits = model(batch)["nuclei"]
        logits_sup = logits[batch["sup_mask"]].squeeze(-1)
        baseline_auroc.update(logits_sup, targets_sup.long())

    base_score = baseline_auroc.compute().item()
    print(f"Baseline AUROC: {base_score:.4f}")

    # 2. Define Feature Groups
    # EFDs: 0-39 | Scale: 40 | Rotation: 41-42
    feature_groups = {
        "EFD_Basic_Shape (Orders 0-2)": slice(0, 12),
        "EFD_Mid_Details (Orders 3-5)": slice(12, 24),
        "EFD_High_Details (Orders 6-9)": slice(24, 40),
        "Scale (log_scales)": slice(40, 41),
        "Rotation (cos/sin)": slice(41, 43),
    }

    importances = {}

    # 3. Iterate and Permute
    for name, f_slice in feature_groups.items():
        perm_auroc = BinaryAUROC().to(device)

        for batch in dataloader:
            batch = to_device(batch)
            targets_sup = batch["labels"]["nuclei"][batch["sup_mask"]]
            if targets_sup.numel() == 0:
                continue

            x_perm = batch["features"].clone()

            # Extract the specific feature slice and shuffle directly along the N_total dimension
            feature_slice = x_perm[:, f_slice]
            shuffled_idx = torch.randperm(feature_slice.size(0), device=device)
            x_perm[:, f_slice] = feature_slice[shuffled_idx]

            # Create a shallow copy of the batch with the permuted features
            permuted_batch = dict(batch)
            permuted_batch["features"] = x_perm

            logits = model(permuted_batch)["nuclei"]
            logits_sup = logits[permuted_batch["sup_mask"]].squeeze(-1)

            perm_auroc.update(logits_sup, targets_sup.long())

        drop = base_score - perm_auroc.compute().item()
        importances[name] = drop
        print(f"  -> Drop after shuffling {name}: {drop:.4f}")

    # 4. Plot Results
    names = list(importances.keys())
    drops = list(importances.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names, drops, color="coral", edgecolor="black")
    ax.set_title("Permutation Feature Importance (Drop in AUROC)")
    ax.set_ylabel("AUROC Decrease")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig("permutation_importance.png", dpi=300)
    plt.close(fig)
    print("Saved permutation_importance.png")


if __name__ == "__main__":
    print("Loading datasets for analysis...")

    slides_df = pd.read_parquet(METADATA_PATH)

    train_df, validation_df = train_test_split(
        slides_df,
        test_size=0.1,
        random_state=42,
        stratify=slides_df["is_carcinoma"],
        groups=slides_df["patient_id"],
    )
    validation_df = validation_df.reset_index(drop=True)
    print(f"Recreated validation split with {len(validation_df)} slides.")

    carcinoma_map = {
        str(k): v for k, v in slides_df.set_index("slide_id")["is_carcinoma"].items()
    }

    sup_dfs = {"annot_labels": pd.read_parquet(ANNOTS_PATH)}

    supervision = build_supervision(
        strategy=SupervisionStrategy(mode="annotation"),
        carcinoma_map=carcinoma_map,
        sup_dfs=sup_dfs,
    )

    dataset = NucleiDataset(
        slides=slides_df,
        supervision=supervision,
        crop_size=4096,
        efd_order=EFD_ORDER,
        full_slide=True,
        predict=False,
        mil=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=4,
        collate_fn=partial(supervised_collate_fn, block_size=128, k=NEIGHBORHOOD_SIZE),
        pin_memory=True,
    )

    compute_permutation_importance(model, dataloader, device)
