import pandas as pd


def compute_slides_positivity(
    df_metadata: pd.DataFrame,
    supervision_mode: str,
    df_annot_labels: pd.DataFrame | None = None,
    df_cam_labels: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Calculates the carcinoma positivity ratio per slide for weighted sampling based on supervision labels.

    Args:
        df_metadata: DataFrame containing a "slide_id" (str) column.
        supervision_mode: One of "annotation", "cam", or "agreement".
        df_annot_labels: Optional DataFrame containing columns "slide_id" (str), "id" (str), and "annot_label" (int).
        df_cam_labels: Optional DataFrame containing columns "slide_id" (str), "id" (str), and "cam_label" (int).

    The `df_annot_labels` and `df_cam_labels` contain only positive slides, negative slides are implicitly considered all-negative.

    Returns:
        A dictionary mapping each slide ID to its fraction of positive nuclei [0, 1].
    """
    assert df_annot_labels is not None or df_cam_labels is not None, (
        "At least one of 'df_annot_labels' or 'df_cam_labels' must be provided."
    )
    positivity_series = pd.Series(dtype=float)

    match supervision_mode:
        case "annotation":
            assert df_annot_labels is not None
            positivity_series = df_annot_labels.groupby("slide_id")[
                "annot_label"
            ].mean()

        case "cam":
            assert df_cam_labels is not None
            confident = df_cam_labels[df_cam_labels["cam_label"] != -1].copy()
            positivity_series = confident.groupby("slide_id")["cam_label"].mean()

        case "agreement":  # positive if both agree on positive
            assert df_annot_labels is not None and df_cam_labels is not None
            merged = df_annot_labels.merge(
                df_cam_labels, on=["slide_id", "id"], how="inner"
            )
            confident = merged[merged["cam_label"] != -1].copy()
            confident["pos_score"] = (
                (confident["annot_label"] == 1) & (confident["cam_label"] == 1)
            ).astype(float)
            positivity_series = confident.groupby("slide_id")["pos_score"].mean()

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
