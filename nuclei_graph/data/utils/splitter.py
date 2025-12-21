import pandas as pd
from sklearn.model_selection import train_test_split


def train_val_split(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split metadata into train and validation sets at the patient level.

    Args:
        metadata: pandas DataFrame containing at least columns: "patient_id" (str),
                  "slide_id" (str), and "slide_nuclei_path" (str).
    """
    patient_labels = metadata.groupby("patient_id")["is_carcinoma"].max()

    train_patients, val_patients = train_test_split(
        patient_labels.index.to_list(),
        test_size=0.1,
        random_state=42,
        stratify=patient_labels.to_list(),
    )
    df_train = metadata[metadata["patient_id"].isin(train_patients)][
        ["slide_id", "slide_nuclei_path"]
    ]
    df_val = metadata[metadata["patient_id"].isin(val_patients)][
        ["slide_id", "slide_nuclei_path"]
    ]
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True)


def get_subset(slide_ids: set[str], df: pd.DataFrame | None) -> pd.DataFrame | None:
    return df[df["slide_id"].isin(slide_ids)] if df is not None else None
