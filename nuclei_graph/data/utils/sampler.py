import pandas as pd


def compute_slides_positivity(
    df_metadata: pd.DataFrame,
    supervision_mode: str,
    df_annot_labels: pd.DataFrame,
    df_cam_labels: pd.DataFrame,
) -> dict[str, float]:
    """Calculates the carcinoma positivity ratio per slide for weighted sampling based on supervision labels.

    The positivity ratio is defined as the fraction of nuclei labeled as positive over the total number of nuclei.

    Args:
        df_metadata: DataFrame containing a "slide_id" (str) column.
        supervision_mode: One of "annotation", "cam", "agreement", "agreement-strict".
        df_annot_labels: DataFrame containing columns "slide_id" (str), "id" (str), and "annot_label" (int).
        df_cam_labels: DataFrame containing columns "slide_id" (str), "id" (str), and "cam_label" (int).

    The `df_annot_labels` and `df_cam_labels` contain only positive slides, negative slides are implicitly considered all-negative.

    Returns:
        A dictionary mapping each slide ID to its fraction of positive nuclei [0, 1].
    """
    positivity_series = pd.Series(dtype=float)

    match supervision_mode:
        case "annotation":
            positivity_series = df_annot_labels.groupby("slide_id")[
                "annot_label"
            ].mean()

        case "cam":
            tmp_cam_labels = df_cam_labels["cam_label"].replace(-1, 0)
            positivity_series = tmp_cam_labels.groupby(df_cam_labels["slide_id"]).mean()

        case "agreement" | "agreement-strict":  # positive if both agree on positive
            tmp_cam_labels = df_cam_labels.copy()
            tmp_cam_labels["cam_label"] = tmp_cam_labels["cam_label"].replace(-1, 0)
            merged = df_annot_labels.merge(
                tmp_cam_labels, on=["slide_id", "id"], how="inner"
            )
            merged["is_positive"] = (
                (merged["annot_label"] == 1) & (merged["cam_label"] == 1)
            ).astype(float)
            positivity_series = merged.groupby("slide_id")["is_positive"].mean()

    positivity_map = df_metadata["slide_id"].map(positivity_series).fillna(0.0)
    return dict(zip(df_metadata["slide_id"], positivity_map, strict=True))


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
