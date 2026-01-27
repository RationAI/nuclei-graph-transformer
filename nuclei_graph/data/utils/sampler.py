import pandas as pd
import pyarrow.parquet as pq


def compute_slides_positivity(
    df_metadata: pd.DataFrame,
    df_labels: pd.DataFrame,
    df_refinement: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Calculates the carcinoma positivity ratio per slide for weighted sampling.

    If "df_refinement" is provided, a nucleus is only considered positive if it is positively
    labeled and is also flagged by the refinement mask (specified in "df_refinement").

    Args:
        df_metadata: DataFrame containing a "slide_id" (str) column.
        df_labels: DataFrame containing columns "slide_id" (str), "id" (str), and "label" (int).
            These exist only for the positive slides, negative slides are implicitly considered all-negative.
        df_refinement: Optional DataFrame containing columns "slide_id" (str), "id" (str), and "refinement_mask" (bool).

    Returns:
        A dictionary mapping each slide ID to its fraction of positive nuclei [0, 1].
    """
    if df_refinement is not None:
        merged = df_labels.merge(df_refinement, on=["slide_id", "id"], how="inner")
        merged["pos_score"] = (merged["label"] * merged["refinement_mask"]).astype(
            "uint8"
        )
        positivity_series = merged.groupby("slide_id")["pos_score"].mean()
    else:
        positivity_series = df_labels.groupby("slide_id")["label"].mean()
    positivity_map = df_metadata["slide_id"].map(positivity_series)
    positivity_map = positivity_map.fillna(0.0)  # negative slides
    return dict(zip(df_metadata["slide_id"], positivity_map, strict=True))


def min_count_filter(df: pd.DataFrame, min_count: int) -> pd.DataFrame:
    """Filter rows in the provided dataframe based on a minimum count of nuclei located at "slide_nuclei_path".

    Args:
        df: Input DataFrame with a "slide_nuclei_path" (str) and "slide_id" (str) columns.
        min_count: Minimum number of nuclei required to retain the slide.
    """
    counts = df["slide_nuclei_path"].apply(
        lambda path: sum(
            fragment.metadata.num_rows for fragment in pq.ParquetDataset(path).fragments
        )
    )
    mask_keep = counts >= min_count
    if not mask_keep.all():
        dropped_slides = df.loc[~mask_keep, ["slide_id"]].copy()
        dropped_slides["nuclei_count"] = counts[~mask_keep]
        print(
            f"[INFO] Dropped slides with < {min_count} nuclei:\n",
            dropped_slides.to_string(index=False),
        )
    return df[mask_keep].reset_index(drop=True)
