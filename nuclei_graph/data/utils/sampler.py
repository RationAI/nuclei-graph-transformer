import pandas as pd


def compute_slides_positivity(
    metadata: pd.DataFrame,
    labels: pd.DataFrame,
    label_mask: pd.DataFrame | None,
) -> dict[str, float]:
    """Calculates the carcinoma positivity ratio per slide for weighted sampling.

    If `label_mask` is provided, a nucleus is only considered positive if it is positively
    labeled and is also flagged by the refinement mask.

    Args:
        metadata: DataFrame containing a "slide_id" (str) column.
        labels: DataFrame containing columns "slide_id" (str), "id" (str), and "label" (int).
        label_mask: Optional DataFrame containing columns "slide_id" (str), "id" (str), and "refinement_mask" (bool).

    Returns:
        A dictionary mapping each slide ID to its fraction of positive nuclei [0, 1].
    """
    if label_mask is not None:
        merged = labels.merge(label_mask, on=["slide_id", "id"], how="inner")
        merged["pos_score"] = (merged["label"] * merged["refinement_mask"]).astype(
            "uint8"
        )
        positivity_series = merged.groupby("slide_id")["pos_score"].mean()
    else:
        positivity_series = labels.groupby("slide_id")["label"].mean()
    positivity_map = metadata["slide_id"].map(positivity_series)
    positivity_map = positivity_map.fillna(0.0)  # negative slides with no annot labels
    return dict(zip(metadata["slide_id"], positivity_map, strict=True))


def pre_crop_filter(metadata: pd.DataFrame, min_count: int) -> pd.DataFrame:
    """Filters out slides that have nuclei count less than `min_count`.

    Used for a cropped-based training to ensure that slides have enough nuclei to sample from.

    Args:
        metadata: DataFrame containing a "slide_nuclei_path" (str) column.
        min_count: Minimum number of nuclei required to retain the slide.
    """
    counts = metadata["slide_nuclei_path"].apply(
        lambda path: pd.read_parquet(path, columns=[]).shape[0]
    )
    mask_keep = counts >= min_count
    if not mask_keep.all():
        dropped_slides = metadata.loc[~mask_keep, ["slide_id"]].copy()
        dropped_slides["nuclei_count"] = counts[~mask_keep]
        print(
            f"[INFO] Dropped slides with < {min_count} nuclei:",
            dropped_slides.to_string(index=False),
        )

    return metadata[mask_keep].reset_index(drop=True)
