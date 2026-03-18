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


def min_positive_nuclei_filter(
    df: pd.DataFrame,
    min_pos_count: float,
    slides_positivity: dict[str, float],
) -> pd.DataFrame:
    """Filter positive slides if their absolute number of positive nuclei is strictly less than `min_pos_count`.

    Args:
        df: Input DataFrame with "slide_id" (str), "nuclei_count" (int) and "is_carcinoma" (bool) columns.
            Should contain "is_carcinoma" (bool) to safely prevent dropping negative slides.
        min_pos_count: Minimum number of positive nuclei required to retain a positive slide.
        slides_positivity: Dictionary mapping slide IDs to their overall positivity ratio.
    """
    positivity_ratios = df["slide_id"].map(slides_positivity)
    abs_pos_count = df["nuclei_count"] * positivity_ratios

    mask_keep = (~df["is_carcinoma"]) | (abs_pos_count >= min_pos_count)

    if not mask_keep.all():
        cols_to_print = ["slide_id", "nuclei_count"]
        dropped_slides = df.loc[~mask_keep, cols_to_print].copy()

        pos_ratios = df.loc[~mask_keep, "slide_id"].map(slides_positivity)
        dropped_slides["pos_nuclei_count"] = (
            (df.loc[~mask_keep, "nuclei_count"] * pos_ratios).round().astype(int)
        )
        print(
            f"[INFO] Dropped slides with < {min_pos_count} positive nuclei:\n",
            dropped_slides.to_string(index=False),
        )

    return df[mask_keep].reset_index(drop=True)
