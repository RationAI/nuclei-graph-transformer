import pandas as pd


def compute_slides_positivity(
    metadata: pd.DataFrame,
    labels: pd.DataFrame,
    label_indicators: pd.DataFrame | None,
) -> dict[str, float]:
    """Calculates the carcinoma positivity ratio per slide for weighted sampling.

    If `label_indicators` is provided, a nucleus is only considered positive if it exists
    within an expert-annotated region and is flagged by the refinement indicator.

    Args:
        metadata: DataFrame containing slide metadata with a "slide_id" (str) column.
        labels: DataFrame containing nuclei labels with "slide_id" (str) and "label" (int) columns.
        label_indicators: Optional DataFrame containing label indicators with "id" (str) and
                          "label_indicator" (bool) columns.

    Returns:
        A dictionary mapping each slide ID to its fraction of positive nuclei [0, 1].
    """
    if label_indicators is not None:
        merged = labels.merge(label_indicators, on="id", how="inner")
        merged["pos_score"] = (merged["label"] * merged["label_indicator"]).astype(
            "uint8"
        )
        positivity_series = merged.groupby("slide_id")["pos_score"].mean()

    else:
        positivity_series = labels.groupby("slide_id")["label"].mean()

    return metadata["slide_id"].map(positivity_series).fillna(0.0).to_dict()


def pre_crop_filter(metadata: pd.DataFrame, min_count: int) -> pd.DataFrame:
    """Filters out slides that have nuclei count less than `min_count`.

    Used for a cropped-based training to ensure that slides have enough nuclei to sample from.

    Args:
        metadata: DataFrame containing slide metadata with a "slide_nuclei_path" (str) column.
        min_count: Minimum number of nuclei required to retain the slide.

    Returns:
        A DataFrame with slides having at least `min_count` nuclei.
    """
    counts = metadata["slide_nuclei_path"].apply(
        lambda path: pd.read_parquet(path, columns=[]).shape[0]
    )
    return metadata[counts >= min_count].reset_index(drop=True)
