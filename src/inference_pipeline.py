### ======================== INFERENCE PIPELINE ================================== ###
"""
THE INFERENCE PIPELINE TAKES EXTERNAL UNSEEN DATA AND ALLOW THE DATA GO THROUGH ALL THE STAGES - 01 - 04
The external data is then ready to access the model and make predictions. Here, the external unseen dataset is the
holdout.csv, this dataset is already cleaned, feature engineered that is, STAGES 01 - 03 is applied already.
Normally, this pipeline would have be developed to account for all those stages and the unseen dataset would have to pass through
all at the inference stage.

We need to import the holdout.csv (feat_eng_wtoVIFholdout.csv and feat_eng_wtVIFholdout.csv) -
since this was what the model was trained on, preprocessors and model ONLY in this inference pipeline.
Note: I am only considering both unpreprocessed/preprocessed holdout datasets, that is, holdout dataset with/without VIF which follows the trained model.
"""

# IMPORT NECESSARY LIBRARIES
import pandas as pd
import os
import joblib

# ============================================================================================= #
# =============== INFERENCE PIPELINE FOR EXTERNAL DATASET WITHOUT VIF PREPROCESSED ============ #
# ============================================================================================= #

# LOADS THE INFERENCE DATASET, PREPROCESSORS AND MODELS (TRAINED_MODEL, FINETUNED_BEST_MODEL)
inference_data_path = "./data/feature_engineered/"
inference_data_name = "feat_eng_wtoVIFholdout.csv"

trained_model_path = "./model/trained_models/"
best_finetuned_model_path = "./model/finetuned_best_models/"

trained_model_name="trained_model_for_unpreprocessed_data.joblib"
best_finetuned_model_name = "best_model.joblib"

# FUNCTION TO LOAD THE INFERENCE DATA
def load_inference_data(data_path: str, data_name: str) -> pd.DataFrame:
    df_path = os.path.join(data_path, data_name)
    return pd.read_csv(df_path)

# FUNCTION TO LOAD PREPROCESSOR AND TRAINED ARTIFACTS
def load_pretrained_artifacts(art_path: str, art_name: str):
    artifact_path = os.path.join(art_path, art_name)
    return joblib.load(artifact_path)


# =============================================== #
# ========= FUNCTION INFERENCE PIPELINE ========= #
# =============================================== #
def inference_pipeline(target_col: str | None = None):
    # LOADING THE INFERENCE DATASET
    df = load_inference_data(inference_data_path, inference_data_name)
    _feature, _target = df.drop(columns=[target_col]), df[target_col]
    
    # LOADING THE PRETRAINED MODELS
    trained_model = load_pretrained_artifacts(trained_model_path, trained_model_name)
    best_fine_tuned_model = load_pretrained_artifacts(best_finetuned_model_path, best_finetuned_model_name)

    # PREDICTING WITH THE MODELS
    predictions = trained_model.predict(_feature)
    trained_pred_df = pd.DataFrame(predictions, columns = ['trained_model_prices'])

    predictions = best_fine_tuned_model.predict(_feature)
    best_pred_df = pd.DataFrame(predictions, columns = ['best_model_prices'])

    # COMPARING ACTUAL WITH PREDICTIONS
    pred_df = pd.concat([_target.reset_index(drop=True), trained_pred_df, best_pred_df], axis=1)

    pred_df['actual_vs_trained'] = pred_df['price'] - pred_df['trained_model_prices']
    pred_df['actual_vs_best'] = pred_df['price'] - pred_df['best_model_prices']

    return pred_df


# ========================================================================== #
# ==== INFERENCE PIPELINE FOR EXTERNAL DATASET WITH VIF PREPROCESSED ======= #
# ========================================================================== #

# LOADS THE INFERENCE DATASET, PREPROCESSORS AND MODEL (TRAINED_MODEL)
inference_data_path_ = "./data/feature_engineered/"
inference_data_name_ = "feat_eng_wtVIFholdout.csv"

preprocessor_path_ = "./model/preproceesed_artifacts/"
trained_model_path_ = "./model/trained_models/"

preprocessor_name_ = "PREPROCESORS.joblib"
trained_model_name_ ="trained_model_for_preproceesed_data.joblib" 

# FUNCTION INFERENCE PIPELINE
def inference_pipeline_with_vif(target_col: str | None = None):
    # LOADING THE INFERENCE DATASET
    df = load_inference_data(inference_data_path_, inference_data_name_)
    _feature, _target = df.drop(columns=[target_col]), df[target_col]
    
    # LOADING THE PREPROCESSORS
    _preprocessor = load_pretrained_artifacts(preprocessor_path_, preprocessor_name_)
    min_max_scaler = _preprocessor['MINMAX_SCALER']
    std_scaler = _preprocessor['STD_SCALER']

    # LOADING THE PRETRAINED MODELS
    trained_model = load_pretrained_artifacts(trained_model_path_, trained_model_name_)
    
    # USING THE PREPROCESSORS
    num_feat = min_max_scaler.transform(_feature)
    scale_feat = std_scaler.transform(num_feat)
    feature = pd.DataFrame(scale_feat, columns=_feature.columns)

    # PREDICTING WITH THE MODELS
    predictions = trained_model.predict(feature)
    trained_pred_df = pd.DataFrame(predictions, columns = ['trained_model_prices'])

    # COMPARING ACTUAL WITH PREDICTIONS
    pred_df = pd.concat([_target.reset_index(drop=True), trained_pred_df], axis=1)

    pred_df['actual_vs_trained'] = pred_df['price'] - pred_df['trained_model_prices']
    
    return pred_df


if __name__ == '__main__':
    price_pred_wto_vif = inference_pipeline('price')
    print(price_pred_wto_vif.head())
    print("="*100)
    price_pred_wt_vif = inference_pipeline_with_vif('price')
    print(price_pred_wt_vif.head())
    print("✅ Inference Pipeline is Successfully for unpreprocessed and preprocessed datasets.")