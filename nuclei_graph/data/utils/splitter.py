import pandas as pd
from sklearn.model_selection import train_test_split


def train_val_split(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits metadata into train and validation sets at the patient level.

    Args:
        metadata: pandas DataFrame containing columns "patient_id" (str) and "is_carcinoma" (bool).
    """
    # patient is considered positive if any of their slides is positive
    patient_labels = metadata.groupby("patient_id")["is_carcinoma"].max()

    train_patients, val_patients = train_test_split(
        patient_labels.index.to_list(),
        test_size=0.1,
        random_state=42,
        stratify=patient_labels.to_list(),
    )
    df_train = metadata[metadata["patient_id"].isin(train_patients)]
    df_val = metadata[metadata["patient_id"].isin(val_patients)]
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True)


def get_subset(df: pd.DataFrame, ids: set[str]) -> pd.DataFrame:
    return df[df["slide_id"].isin(ids)]
