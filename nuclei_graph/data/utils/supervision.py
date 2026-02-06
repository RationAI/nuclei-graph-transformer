import pandas as pd
import torch
from tqdm import tqdm

from nuclei_graph.nuclei_graph_typing import DatasetSupervision, SlideSupervision


def build_supervision(
    df_annot: pd.DataFrame, df_cam: pd.DataFrame, label_map: dict[str, int]
) -> DatasetSupervision:
    """Packs annotation and CAM labels into a DatasetSupervision dataclass containing Tensors **sorted** by slide_id and id.

    Args:
        df_annot: DataFrame containing annotation labels with columns "slide_id" (str), "id" (str), and "annot_label" (int).
        df_cam: DataFrame containing CAM labels with columns "slide_id" (str), "id" (str), and "cam_label" (int).
        label_map: Mapping from slide_id to slide-level label (int).
    """
    df_annot = df_annot.sort_values(["slide_id", "id"])
    df_cam = df_cam.sort_values(["slide_id", "id"])

    annot_groups = dict(list(df_annot.groupby("slide_id")))
    cam_groups = dict(list(df_cam.groupby("slide_id")))

    sup_map = {}

    for slide_id, label in tqdm(label_map.items(), desc="Building Supervision"):
        annot = cam = None

        if slide_id in annot_groups:
            group_annot = annot_groups[slide_id]
            group_cam = cam_groups[slide_id]

            annot = torch.tensor(group_annot["annot_label"].values, dtype=torch.float32)
            cam = torch.tensor(group_cam["cam_label"].values, dtype=torch.float32)

        sup_map[slide_id] = SlideSupervision(
            slide_label=int(label),
            annot_labels=annot,
            cam_labels=cam,
        )

    return DatasetSupervision(sup_map=sup_map)
