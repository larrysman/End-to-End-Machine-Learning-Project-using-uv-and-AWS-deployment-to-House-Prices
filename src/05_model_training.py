### =================== STAGE 05 - MODEL TRAINING PIPELINE ====================== ###

"""
MODULES TO MODEL TRAINING FOR BOTH DATASETS
- Loads datasets from preprocessed folder ../data/preprocessed
- Loads datasets from feature_engineered folder ../data/feature_engineered
- Train and evaluate the performance of the models using metrics (MSE, RMSE, MAE, R2)
- Save the trained models and metrics.
"""

# IMPORT NECESSARY LIBRARIES
import numpy as np
import pandas as pd
import os
from typing import Optional
from xgboost import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# LOADS THE DATASETS FOR PREPROCESSED
train_prep = pd.read_csv("./data/preprocessed/train_preprocessed.csv")
eval_prep = pd.read_csv("./data/preprocessed/eval_preprocessed.csv")

# MODEL TRAINING FOR PREPROCESSING DATASETS
def model_training_for_prep_data(df1: pd.DataFrame, df2: pd.DataFrame, target_col: str, model_params: Optional[dict]=None, random_state: int = 42):
    train_df = df1.copy()
    eval_df = df2.copy()

    X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
    X_eval, y_eval = eval_df.drop(columns=[target_col]), eval_df[target_col]

    params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist",
        "random_state": random_state,
        "n_jobs": -1
    }

    if model_params:
        params.update(model_params)

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_eval)
    mse = float(mean_squared_error(y_eval, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_eval, y_pred))
    r2 = float(r2_score(y_eval, y_pred))

    METRICS = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2_SCORE": r2
    }

    # CREATING THE OUTPUT FOLDERS TO STORE THE OUTCOMES OF THIS STAGE
    model_path = os.path.join(os.getcwd(), "model", "trained_models")
    os.makedirs(model_path, exist_ok=True)
    metric_path = os.path.join(os.getcwd(), "model", "model_metrics")
    os.makedirs(metric_path, exist_ok=True)

    # SAVING THE OUTCOMES FILE AND SCALERS INTO THE CREATED FOLDERS
    joblib.dump(
        model,
        os.path.join(model_path, "trained_model_for_preproceesed_data.joblib"),
        compress=3
    )

    joblib.dump(
        METRICS,
        os.path.join(metric_path, "metrics_for_prep_data.joblib"),
        compress=3
    )

    print("The trained model and metrics for preprocessing data are: ")
    print(f"MEAN_SQUARED_ERROR:  {mse:,.2f}")
    print(f"ROOT_MEAN_SQUARED_ERROR:  {rmse:,.2f}")
    print(f"MEAN_ABSOLUTE_ERROR:  {mae:,.2f}")
    print(f"COEFFICIENT_OF_DETERMINATION: {r2*100:,.2f}%")
    print("✅ Model Trained Successfully for preprocessed datasets with model and metrics saved.")

    return model, METRICS


""" MODEL TRAINING TO BE FINETUNED AND DEPLOYED """
# LOADS THE DATASETS WITHOUT PREPROCESSED (NOT REMOVING MULTICOLLINEARITY AND NOT SCALING)
train_ = pd.read_csv("./data/feature_engineered/feat_eng_wtoVIFtrained.csv")
eval_ = pd.read_csv("./data/feature_engineered/feat_eng_wtoVIFeval.csv")

# MODEL TRAINING FOR PREPROCESSING DATASETS
def model_training_without_preprocessing(df1: pd.DataFrame, df2: pd.DataFrame, target_col: str, model_params: Optional[dict]=None, random_state: int = 42):
    train_df = df1.copy()
    eval_df = df2.copy()

    X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
    X_eval, y_eval = eval_df.drop(columns=[target_col]), eval_df[target_col]

    params = {
        "n_estimators": 500,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": random_state,
        "n_jobs": -1
    }

    if model_params:
        params.update(model_params)

    model = ExtraTreesRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_eval)
    mse = float(mean_squared_error(y_eval, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_eval, y_pred))
    r2 = float(r2_score(y_eval, y_pred))

    METRICS = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2_SCORE": r2
    }

    # CREATING THE OUTPUT FOLDERS TO STORE THE OUTCOMES OF THIS STAGE
    model_path = os.path.join(os.getcwd(), "model", "trained_models")
    os.makedirs(model_path, exist_ok=True)
    metric_path = os.path.join(os.getcwd(), "model", "model_metrics")
    os.makedirs(metric_path, exist_ok=True)

    # SAVING THE OUTCOMES FILE AND SCALERS INTO THE CREATED FOLDERS
    joblib.dump(
        model,
        os.path.join(model_path, "trained_model_for_unpreprocessed_data.joblib"),
        compress=3
    )

    joblib.dump(
        METRICS,
        os.path.join(metric_path, "metrics_for_unpreprocessed_data.joblib"),
        compress=3
    )

    print("The trained model and metrics for unpreprocessed data are: ")
    print(f"MEAN_SQUARED_ERROR:  {mse:,.2f}")
    print(f"ROOT_MEAN_SQUARED_ERROR:  {rmse:,.2f}")
    print(f"MEAN_ABSOLUTE_ERROR:  {mae:,.2f}")
    print(f"COEFFICIENT_OF_DETERMINATION: {r2*100:,.2f}%")
    print("✅ Model Trained Successfully for unpreprocessed datasets with model and metrics saved.")

    return model, METRICS

model_training_without_preprocessing(train_, eval_, "price")

if __name__ == "__main__":
    model_prep, METRICS_prep = model_training_for_prep_data(train_prep, eval_prep, "price")
    model_wto, METRICS_wto = model_training_without_preprocessing(train_, eval_, "price")
