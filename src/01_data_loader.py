### ========================= STAGE 01 - DATA LOADING AND SPLITTING MODULE =========================== ###
"""
FUNCTIONS TO LOAD RAW DATA, CONVERT DATE TO STANDARD DATETIME, SPLIT DATASETS INTO TRAINING, EVALUATION AND HOLDOUT. SAVE THE SPLITTED DATASETS INTO A NEW FOLDER (../DATA/RAW).
"""
# IMPORT RELEVANT LIBRARIES
import pandas as pd
import os
import config
from typing import Optional

# PATH TO THE RAW DATA
raw_data_and_path = config.RAW_DATA_PATH["PATH"]

# LOAD RAW DATA FROM FILE
def load_data(data_path: str = raw_data_and_path) -> pd.DataFrame:
    return pd.read_csv(data_path)

def data_sample(df: pd.DataFrame, sample_frac: Optional[float], random_state: int=42) -> pd.DataFrame:
    if sample_frac is None:
        return df
    sample_frac = float(sample_frac)
    if sample_frac <= 0 or sample_frac >= 1:
        return df
    print(f"✅ {sample_frac*100}% of the raw dataset is loaded for further analysis.")
    return df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)

# DATE CONVERSION TO DATETIME OBJECT
def convert_date_column(date_column: str = "date") -> pd.DataFrame:
    df = load_data()
    df = data_sample(df, sample_frac=0.6)
    df[date_column] = pd.to_datetime(df[date_column])
    print(f"The starting date is: {df[date_column].min()}")
    print(f"The ending date is: {df[date_column].max()}")
    df = df.sort_values(by=date_column)
    df.reset_index(drop=True, inplace=True)
    return df

# DATA SPLITTING FUNCTION FOR TEMPORAL DATA
def splitting_data(date_column: str = "date") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # CUT-OFF DATES
    cutoff_train = "2019-12-31"   # Training data ends 2019-12-31
    cutoff_eval = "2021-12-31"    # Evaluation data ends 2021-12-31
    cutoff_holdout = "2022-01-01" # Holdout data starts 2022-01-01

    df = convert_date_column()

    # TRAINING SET: 2012 - 2019
    train_df = df[df[date_column] <= cutoff_train]
    # EVALUATION/VALIDATION/TEST SET: 2020 - 2021
    eval_df = df[(df[date_column] > cutoff_train) & (df[date_column] <= cutoff_eval)]
    # HOLDOUT SET: 2022 - 2023
    holdout_df = df[df[date_column] >= cutoff_holdout]

    return train_df, eval_df, holdout_df

# SAVING THE OUTCOMES OF THIS STAGE
def saving_output_files() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, eval_df, holdout_df = splitting_data()

    # CREATING THE OUTPUT FOLDERS TO STORE THE OUTCOMES OF THIS STAGE
    output_path = os.path.join(os.getcwd(), "data", "raw")
    os.makedirs(output_path, exist_ok=True)

    # SAVING THE OUTCOMES FILE INTO THE CREATED FOLDERS
    train_df.to_csv(os.path.join(output_path, "train.csv"), index=False)
    eval_df.to_csv(os.path.join(output_path, "eval.csv"), index=False)
    holdout_df.to_csv(os.path.join(output_path, "holdout.csv"), index=False)

    print(f"✅ Raw data loaded, date converted, data splitted into train, eval and holdout, and saved to {output_path}.")
    print(f"  Training: {train_df.shape}, Evaluation: {eval_df.shape}, Holdout: {holdout_df.shape}")

    return train_df, eval_df, holdout_df


if __name__ == "__main__":
    print("Running data loader module...")
    train_df, eval_df, holdout_df = saving_output_files()