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
            f"[INFO] Dropped slides with < {min_count} nuclei:\n"
            f"{dropped_slides.to_string(index=False)}"
        )
    return df[mask_keep].reset_index(drop=True)


def min_positive_count_filter(
    df: pd.DataFrame,
    min_pos_count: float,
    pos_counts: dict[str, int],
) -> pd.DataFrame:
    """Filter positive slides if their absolute number of positive nuclei is strictly less than `min_pos_count`.

    Args:
        df: Input DataFrame with "slide_id" and "is_carcinoma" columns.
        min_pos_count: Minimum absolute number of positive nuclei required to retain a positive slide.
        pos_counts: Dictionary mapping slide IDs to their count of confident positive nuclei.
    """
    pos_count = df["slide_id"].map(pos_counts)
    mask_keep = (~df["is_carcinoma"]) | pos_count.ge(min_pos_count)

    if not mask_keep.all():
        dropped_slides = df.loc[~mask_keep, ["slide_id", "nuclei_count"]].copy()
        dropped_slides["pos_nuclei_count"] = pos_count[~mask_keep]

        print(
            f"[INFO] Dropped slides with < {min_pos_count} positive nuclei:\n"
            f"{dropped_slides.to_string(index=False)}"
        )

    return df[mask_keep].reset_index(drop=True)
