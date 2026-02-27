import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def get_distribution(labels: pd.Series) -> pd.Series:
    return labels.value_counts(normalize=True).sort_index()


def train_val_split(
    metadata: pd.DataFrame, test_size: float = 0.1, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits metadata into train and validation sets at the patient level, stratifying by slide label.

    Source: Prostate Cancer repository.
    """
    data = metadata.copy().sort_values(by="slide_id").reset_index(drop=True)

    labels = data["is_carcinoma"]
    groups = data["patient_id"]

    n_splits = round(1 / test_size)
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    data_distribution = get_distribution(labels)
    min_diff = float("inf")
    train_idx, val_idx = None, None

    for curr_train_idx, curr_val_idx in sgkf.split(X=data, y=labels, groups=groups):
        val_distribution = get_distribution(labels.iloc[curr_val_idx])
        diff = (val_distribution - data_distribution).abs().sum()

        if len(val_distribution) != len(data_distribution):
            diff = float("inf")

        if diff < min_diff:
            min_diff = diff
            train_idx = curr_train_idx
            val_idx = curr_val_idx

    assert train_idx is not None and val_idx is not None

    df_train = data.iloc[train_idx].reset_index(drop=True)
    df_val = data.iloc[val_idx].reset_index(drop=True)

    # --- Summary ---
    print("\n" + "DATA SPLIT SUMMARY")

    orig_pos_rate = labels.mean()
    print(f"Original Dataset Positivity: {orig_pos_rate:.1%}\n")

    for name, split_df in [("Train", df_train), ("Val", df_val)]:
        n_slides = len(split_df)
        n_patients = split_df["patient_id"].nunique()
        pos_slides = split_df["is_carcinoma"].sum()
        neg_slides = n_slides - pos_slides
        pos_rate = pos_slides / n_slides if n_slides > 0 else 0

        print(f"{name} Set:")
        print(f"  Patients:   {n_patients}")
        print(f"  Slides:     {n_slides} (Pos: {pos_slides}, Neg: {neg_slides})")
        print(f"  Positivity: {pos_rate:.1%}")
        print("-" * 30 + "\n")
    # ---------------------------

    return df_train, df_val


def get_subset(df: pd.DataFrame, ids: set[str]) -> pd.DataFrame:
    return df[df["slide_id"].isin(ids)]
