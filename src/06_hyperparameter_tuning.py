### ============================= STAGE 06 - HYPERPARAMETER TUNING WITH OPTUNA ========================= ###
"""
MODULES TO PERFORM HYPERPARAMETER TUNING OF THE BASELINE MODEL
- Loads datasets from feature_engineered folder ../data/feature_engineered
- Loads the pre-trained model from ../model/trained_models/trained_model_for_unpreprocessed_data.joblib
- Register the pretrained in MLFLOW.
- Retraining the model by optimizing the hyperparameters with optuna.
- Select the best hyperparameters and log to MLFLOW.
- Retraining the model with the best hyperparameters and evaluate the performance using metrics (MSE, RMSE, MAE, R2)
- Save the optimized model and metrics.
"""

# IMPORT NECESSARY LIBRARIES 
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import joblib
from typing import Optional
import mlflow
import mlflow.sklearn
import optuna

# LOADS THE DATASET
train_df = pd.read_csv("./data/feature_engineered/feat_eng_wtoVIFtrained.csv")
eval_df = pd.read_csv("./data/feature_engineered/feat_eng_wtoVIFeval.csv")
model_path="./model/trained_models/"
model_name="trained_model_for_unpreprocessed_data.joblib"
metric_path="./model/model_metrics/"
metric_name="metrics_for_unpreprocessed_data.joblib"

# LOAD THE PRE-TRAINED MODEL
def load_pretrained_model(model_path: str, model_name: str):
    artifact_path = os.path.join(model_path, model_name)
    model = joblib.load(artifact_path)
    """print(model.get_params())"""
    return model

# LOAD THE PRE-TRAINED METRICS
def load_pretrained_metrics(metric_path: str, metric_name: str):
    artifact_path = os.path.join(metric_path, metric_name)
    METRICS = joblib.load(artifact_path)
    # CEHCKS AND VALIDATION
    assert isinstance(METRICS, dict)
    assert "MSE" in METRICS
    assert "RMSE" in METRICS
    assert "MAE" in METRICS
    assert "R2_SCORE" in METRICS
    # ACCESSING METRICS
    MAE = METRICS["MAE"]
    R2 = METRICS["R2_SCORE"]
    print(f"MAE: {round(MAE, 2)} and R2_SCORE: {round(R2*100, 2)}%")

    return MAE, R2

# LOGGING THE BASELINE MODEL AND METRICS INTO MLFLOW
def baseline_model_mlfow_log(model_path: str, model_name: str, metric_path: str, metric_name: str, tracking_uri: Optional[str]=None, exp_name: str="baseline_model_logging"):
    baseline_model = load_pretrained_model(model_path, model_name)
    MAE, R2 = load_pretrained_metrics(metric_path, metric_name)

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(exp_name)

    with mlflow.start_run(run_name="baseline_model"):
        mlflow.sklearn.log_model(baseline_model, name="baseline_model")
        mlflow.log_metrics({"mae": round(MAE,2), "r2_score": round(R2*100,2)})

        run_id = mlflow.active_run().info.run_id
    print(f"The running id for the baseline model logging is: {run_id}")
    return run_id

"""baseline_model_mlfow_log(
    model_path="./model/trained_models/", model_name="trained_model_for_unpreprocessed_data.joblib",
    metric_path=".\model\model_metrics", metric_name="metrics_for_unpreprocessed_data.joblib"
    )"""

# RETRAINING THE MODEL BY OPTIMIZING AND FINE-TUNING THE HYPERPARAMTERS WITH OPTUNA 
def _data_splitter(df1: pd.DataFrame, df2: pd.DataFrame, target_col: str = "price") -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    train_ = df1.copy()
    eval_ = df2.copy()

    feature_cols = [col for col in train_.columns if col != target_col]
    X_train, y_train = train_[feature_cols], train_[target_col]
    X_eval, y_eval = eval_[feature_cols], eval_[target_col]
    return X_train, y_train, X_eval, y_eval

"""
FINE-TUNING WITH OPTUNA
"""
def finetune_model(df1: pd.DataFrame, df2: pd.DataFrame, n_trials: int, exp_name: str, tracking_uri: Optional[str]=None) -> tuple[dict, dict]:
    """ Run Optuna tuning: save the best model; return (best_params, best_metrics)."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name=exp_name)

    X_train, y_train, X_eval, y_eval = _data_splitter(df1, df2)

    def optuna_objective(trial: optuna.Trial):
        # DEFINE THE HYPERPARAMETER SEARCH SPACE
        PARAMS = {
            "n_estimators": trial.suggest_int("n_estimators", 300, 800),
            "max_depth": trial.suggest_categorical("max_depth", [None, 10, 15, 20, 30]),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 15),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
            "criterion": "squared_error",
            "bootstrap": False,
            "ccp_alpha": trial.suggest_float("ccp_alpha", 0.0, 0.02),
            "random_state": 42,
            "n_jobs": -1
        }

        with mlflow.start_run(nested=True):
            mlflow.log_params(PARAMS)

            MODEL = ExtraTreesRegressor(**PARAMS)
            MODEL.fit(X_train, y_train)

            preds = MODEL.predict(X_eval)

            mse = float(mean_squared_error(y_eval, preds))
            rmse = float(np.sqrt(mse))
            mae = float(mean_absolute_error(y_eval, preds))
            r2 = float(r2_score(y_eval, preds))

            mlflow.log_metrics({"MSE": mse, "RMSE": rmse, "MAE": mae, "R2_SCORE": r2})
        return mae
    
    # RUNNING OPTUNA STUDY WITH TRACKING IN MLFLOW
    study = optuna.create_study(direction="minimize", study_name="Baseline_Model_Optimization")
    study.optimize(optuna_objective, n_trials=n_trials)

    best_trial = study.best_trial
    best_params = best_trial.params
    print(f"✅ Baseline Model Best Trial Parameters are: {best_params}")

    # RETRAINING WITH THE BEST TRIAL PARAMETERS AND LOG INTO MLFLOW
    optimized_model = ExtraTreesRegressor(**{**best_params, "random_state": 42, "n_jobs": -1})
    optimized_model.fit(X_train, y_train)

    preds = optimized_model.predict(X_eval)
    optimized_metrics = {
        "MSE": float(mean_squared_error(y_eval, preds)),
        "RMSE": float(np.sqrt(mean_squared_error(y_eval, preds))),
        "MAE": float(mean_absolute_error(y_eval, preds)),
        "R2_SCORE": float(r2_score(y_eval, preds))
    }
    print(f"✅ Best tuned model metrics: {optimized_metrics}")

    # SAVING THE BEST MODEL AND METRICS
    # CREATING THE OUTPUT FOLDERS TO STORE THE OUTCOMES OF THIS STAGE
    model_path = os.path.join(os.getcwd(), "model", "finetuned_best_models")
    os.makedirs(model_path, exist_ok=True)
    metric_path = os.path.join(os.getcwd(), "model", "finetuned_best_metrics")
    os.makedirs(metric_path, exist_ok=True)

    # SAVING THE OUTCOMES FILE AND SCALERS INTO THE CREATED FOLDERS
    joblib.dump(
        optimized_model,
        os.path.join(model_path, "best_model.joblib"),
        compress=3
    )

    joblib.dump(
        optimized_metrics,
        os.path.join(metric_path, "best_metrics.joblib"),
        compress=3
    )

    print("✅ Best tuned model and metrics saved.")

    # LOGGING FINAL BEST MODEL, PARAMETERS AND METRICS INTO MLFLOW
    with mlflow.start_run(run_name="best_extratreesregressor_model"):
        mlflow.log_params(best_params)
        mlflow.log_metrics(optimized_metrics)
        mlflow.sklearn.log_model(optimized_model, name="model")

    return best_params, optimized_metrics

if __name__ == "__main__":
    baseline_model_mlfow_log(model_path, model_name, metric_path, metric_name)
    finetune_model(train_df, eval_df, n_trials=10, exp_name="EXRATREESREGRESSOR_OPTUNA_FOR_HOUSING_PRICE")