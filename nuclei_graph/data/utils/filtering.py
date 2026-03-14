import pandas as pd


def min_count_filter(df: pd.DataFrame, min_count: int) -> pd.DataFrame:
    """Filter rows in the provided dataframe based on a minimum count of nuclei located at "slide_nuclei_path".

    Args:
        df: Input DataFrame with a "slide_nuclei_path" (str), "slide_id" (str), and "nuclei_count" (int) columns.
        min_count: Minimum number of nuclei required to retain the slide.
    """
    mask_keep = df["nuclei_count"] >= min_count
    if not mask_keep.all():
        dropped_slides = df.loc[~mask_keep, ["slide_id", "nuclei_count"]].copy()
        print(
            f"[INFO] Dropped slides with < {min_count} nuclei:\n",
            dropped_slides.to_string(index=False),
        )
    return df[mask_keep].reset_index(drop=True)
