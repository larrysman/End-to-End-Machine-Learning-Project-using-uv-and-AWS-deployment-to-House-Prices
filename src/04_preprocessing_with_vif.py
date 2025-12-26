### ====================== STAGE 04 - PREPROCESSING PIPELINE FOR NUMERICAL AND CATEGORICAL FEATURES AFTER REMOVING MULTICOLLINEARITY ============= ###

"""
MODULES TO PREPROCESSED FEATURES NECESSARY FOR MODEL TRAINING
- Loads feat_eng_wtVIFtrain/eval datasets from ../data/feature_engineered
- Scaled using StandardScaler and MinMaxScaler since datasets contains only the numerical features.
- Save the datasets preprocessed to ../data/preprocessedwtVIF.
"""

# IMPORT NECESSARY LIBRARIES
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import joblib

# LOAD THE DATASETS
train_df = pd.read_csv("./data/feature_engineered/feat_eng_wtVIFtrained.csv")
eval_df = pd.read_csv("./data/feature_engineered/feat_eng_wtVIFeval.csv")

# SPLITTING DATA INTO X_TRAIN, Y_TRAIN, X_EVAL AND Y_EVAL
def splitting_into_train_test(df1: pd.DataFrame, df2: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:

    feature_cols = [col for col in df1.columns.intersection(df2.columns) if col != target_col]

    X_train = df1[feature_cols]
    y_train = df1[target_col]

    X_eval = df2[feature_cols]
    y_eval = df2[target_col]

    return X_train, y_train, X_eval, y_eval

# DATASET STANDARDIZATION AND MINIMUM_MAXIMUM SCALING
def scaling_features(train_df: pd.DataFrame, eval_df: pd.DataFrame, target_col: str):
    # INSTANTIATE THE SCALERS
    min_max_scaler = MinMaxScaler()
    std_scaler = StandardScaler()
    # DATASETS SPLIT INTO TRAINING AND TESTING (EVALUATION)
    X_train, y_train, X_eval, y_eval = splitting_into_train_test(train_df, eval_df, target_col)
    # FITTING AND TRANSFORM
    X_train_min_max = min_max_scaler.fit_transform(X_train)
    X_train_scaled = std_scaler.fit_transform(X_train_min_max)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    train_df = pd.concat([X_train_scaled, y_train.reset_index(drop=True)], axis=1)

    X_eval_min_max = min_max_scaler.transform(X_eval)
    X_eval_scaled = std_scaler.transform(X_eval_min_max)
    X_eval_scaled =pd.DataFrame(X_eval_scaled, columns=X_eval.columns)
    eval_df = pd.concat([X_eval_scaled, y_eval.reset_index(drop=True)], axis=1)
    print(f"✅ All training and evaluation datasets are fully preprocessed with minmax and standard scalers.")
    return train_df, eval_df, min_max_scaler, std_scaler

# RUNNING THE PREPROCESSING AND SAVING ALL PREPROCESSED DATASETS AND PREPROCESSORS
def run_preprocess(train_df: pd.DataFrame, eval_df: pd.DataFrame, target_col: str):

    train_df, eval_df, min_max_scaler, std_scaler = scaling_features(train_df, eval_df, target_col)
    # CREATING THE OUTPUT FOLDERS TO STORE THE OUTCOMES OF THIS STAGE
    output_path = os.path.join(os.getcwd(), "data", "preprocessed")
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(os.getcwd(), "model", "preproceesed_artifacts")
    os.makedirs(model_path, exist_ok=True)

    # SAVING THE OUTCOMES FILE AND SCALERS INTO THE CREATED FOLDERS
    train_df.to_csv(os.path.join(output_path, "train_preprocessed.csv"), index=False)
    eval_df.to_csv(os.path.join(output_path, "eval_preprocessed.csv"), index=False)

    PREPROCESSOR = {
        "MINMAX_SCALER": min_max_scaler,
        "STD_SCALER": std_scaler
    }

    joblib.dump(
        PREPROCESSOR,
        os.path.join(model_path, "PREPROCESORS.joblib"),
        compress=3
    )

    print(f"✅ Scaling and Standardization on the engineered data for training and evaluation datasets have been successfully saved.")
    print(f"✅ Preprocessors have been saved successfully.")

if __name__ == "__main__":
    run_preprocess(train_df=train_df, eval_df=eval_df, target_col="price")
