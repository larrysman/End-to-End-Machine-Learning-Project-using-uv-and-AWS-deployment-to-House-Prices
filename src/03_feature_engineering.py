### ============================= STAGE 03 - FEATURE ENGINEERING MODULE ============================ ###
"""
MODULES TO ENGINEERED FEATURES NECESSARY FOR MODEL PERFORMANCE
- Loads train/eval/holdout datasets from ../data/cleaned
- Engineered features from dates.
- Test for multicollinearity using variance_inflation_factor (VIF).
- Drop all features with higher VIF (>5).
- Compute the correlation analysis for features with the target.
- Drop all highly correlated features with the target.
- Engineered features from Numerical and Categorical columns with focus on (zip_code and city_full).
- Drop all unwanted columns.
- Save the datasets (both with removing and without removing multicollinear features) to ..data/feature_engineered.
"""

# IMPORT NECESSARY LIBRARIES
import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sb
from statsmodels.stats.outliers_influence import variance_inflation_factor
from category_encoders import TargetEncoder

# LOADS ALL THE DATASETS
train = pd.read_csv("./data/cleaned/train_cleaned.csv")
eval_df = pd.read_csv("./data/cleaned/eval_cleaned.csv")
holdout = pd.read_csv("./data/cleaned/holdout_cleaned.csv")

# FUNCTION THAT ENGINEER DATE FEATURES
def engineer_date_features(df: pd.DataFrame, col_name: str = "date") -> pd.DataFrame:

    df[col_name] = pd.to_datetime(df[col_name], errors="coerce")
    df["month"] = df[col_name].dt.month
    df["quarter"] = df[col_name].dt.quarter
    df["year"] = df[col_name].dt.year
    # DROP THE ORIGINAL DATE COLUMN
    df = df.drop(columns=[col_name])
    # REORDERING COLUMNS TO PLACE THE NEW DATE FEATURES AT THE FRONT
    df.insert(0, "month", df.pop("month"))
    df.insert(1, "quarter", df.pop("quarter"))
    df.insert(2, "year", df.pop("year"))
    return df

# FUNCTION TO COMPUTE THE VARIANCE INFLATION FACTOR TO DETERMINE COLUMNS TO BE REMOVED WITH MULTICOLLINEARITY
def compute_variance_inflation_factor(df: pd.DataFrame, target_col: str="price", vif_threshold: float | None = None) -> list[str]:
    """
    The function will return list of columns with high variance inflation factor depending on the VIF number indicated.

    Arguments:
    - df: DataFrame
    - target: defined at default to be price for the work.
    - VIF no: specify the number of VIF you want to consider.

    Returns:
    - List: This contains all columns with higher VIF number indicating multicollinearity.
    """
    features_df = df.drop(columns=[target_col])
    num_features_df = features_df.select_dtypes(include=[np.number])
    vif_data_df = pd.DataFrame()
    vif_data_df["num_features_cols"] = num_features_df.columns
    vif_data_df["VIF_values"] = [round(variance_inflation_factor(num_features_df.values, k),0) for k in range(num_features_df.shape[1])]
    vif_data_df = vif_data_df.sort_values(by="VIF_values", ascending=False).reset_index(drop=True)
    # THE MULTICOLLINEARITY COLUMNS TO BE REMOVED
    OFFSET_COLUMNS = vif_data_df[vif_data_df["VIF_values"] >= vif_threshold]["num_features_cols"].tolist()
    return OFFSET_COLUMNS

# OPTIMIZED FUNCTION TO COMPUTE THE VARIANCE INFLATION FACTOR TO DETERMINE COLUMNS TO BE REMOVED WITH MULTICOLLINEARITY
def compute_variance_inflation_factor_optimized(df: pd.DataFrame, target_col: str="price", vif_threshold: float | None = None) -> list[str]:

    """
    The function will return list of columns with high variance inflation factor depending on the VIF number indicated.

    Arguments:
    - df: DataFrame
    - target: defined at default to be price for the work.
    - VIF no: specify the number of VIF you want to consider.

    Returns:
    - List: This contains all columns with higher VIF number indicating multicollinearity.
    """

    if target_col not in df.columns:
        raise ValueError(f"The target column '{target_col}' is not found in the dataset.")
    features_df = df.drop(columns=[target_col])
    num_features_df = features_df.select_dtypes(include=[np.number])

    if num_features_df.shape[1] < 2:
        return []
    
    vif_data_df = pd.DataFrame({
        "num_features_cols": num_features_df.columns,
        "VIF_values": [
            variance_inflation_factor(num_features_df.values, k)
            for k in range(num_features_df.shape[1])
        ]
    }).sort_values(by="VIF_values", ascending=False)

    if vif_threshold is None:
        return vif_data_df["num_features_cols"].tolist()
    
    return vif_data_df.loc[vif_data_df["VIF_values"] >= vif_threshold, "num_features_cols"].tolist()

# CORRELATION ANALYSIS WITH THE TARGET AND VISUALIZATION
def features_correlation_with_target(df: pd.DataFrame, target_col: str=None):
    
    num_features = df.select_dtypes(include=[np.number])

    if target_col not in num_features.columns:
        raise ValueError(f"Target column '{target_col}' is not a numerical feature in the dataframe.")
    
    corr_with_target = num_features.corr(method='pearson')[target_col].drop(target_col).sort_values(ascending=False)

    sb.set_theme(style="whitegrid")
    sb.set(font_scale=1.2)
    plt.figure(figsize=(15, 7))
    ax = sb.barplot(x=corr_with_target.index, y=corr_with_target.values, palette="viridis")
    plt.xticks(rotation=90)
    plt.title(f"CORRELATION OF NUMERICAL FEATURES WITH THE TARGET -->> {target_col}", fontsize=16)
    plt.xlabel("NUMERICAL FEATURES", fontsize=14)
    plt.ylabel("PEARSON CORRELATION COEFFICIENT (r)", fontsize=14)
    plt.axhline(0, color='red', linestyle='--')
    plt.tight_layout()
    plt.show()

def features_correlation_with_target(df: pd.DataFrame, target_col: str=None):
    
    num_features = df.select_dtypes(include=[np.number])

    if target_col not in num_features.columns:
        raise ValueError(f"Target column '{target_col}' is not a numerical feature in the dataframe.")
    
    corr_with_target = num_features.corr(method='pearson')[target_col].drop(target_col).sort_values(ascending=False)

    sb.set_theme(style="whitegrid")
    sb.set(font_scale=1.2)
    plt.figure(figsize=(15, 7))
    ax = sb.heatmap(
        corr_with_target.to_frame(),
        annot=True,
        fmt=".2f",
        cmap=sb.diverging_palette(220, 20, as_cmap=True),
        center=0,
        cbar_kws={"shrink": 0.8, "pad":0.02},
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        linecolor='black'
    )
    ax.set_title(f"CORRELATION OF NUMERICAL FEATURES WITH THE TARGET -->> {target_col}", fontsize=16)
    ax.set_xlabel("CORRELATION COEFFICIENT WITH THE TARGET", fontsize=14)
    ax.set_ylabel("NUMERICAL FEATURES", fontsize=14)
    plt.tight_layout()
    plt.show()

# FUNCTION TO DROP THE CORRELATED COLUMNS
def removing_correlated_columns(
        df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame,
        cols: list = ["median_sale_price", "Median Home Value", "median_list_ppsf", "supermarket", "bank"]
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df1.copy()
    eval_df = df2.copy()
    holdout_df = df3.copy()

    train_df = train_df.drop(columns=cols)
    eval_df = eval_df.drop(columns=cols)
    holdout_df = holdout_df.drop(columns=cols)
    return train_df, eval_df, holdout_df

# ENGINEERING OTHER FEATURES WITH NUMERICAL AND CATEGORICAL COLUMNS
"""ZIPCODE FEATURE ENGINEERING"""
def engineering_zipcode_with_frequency_encoding(df: pd.DataFrame, col_zip: str="zipcode") -> pd.DataFrame:
    zipcode_counts = df[col_zip].value_counts()
    df[f"{col_zip}_freq"] = df[col_zip].map(zipcode_counts)
    df[f"{col_zip}_freq"] = df[f"{col_zip}_freq"].fillna(0)
    return df

"""CITY_FULL FEATURE ENGINEERING"""
def engineering_cityfull_with_target_encoder(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame, col: str="city_full", target: str="price"):

    train_df = df1.copy()
    eval_df = df2.copy()
    holdout_df = df3.copy()

    target_encoder = TargetEncoder(cols=[col])
    train_df[f"{col}_encoded"] = target_encoder.fit_transform(train_df[col], train_df[target])
    eval_df[f"{col}_encoded"] = target_encoder.transform(eval_df[col])
    holdout_df[f"{col}_encoded"] = target_encoder.transform(holdout_df[col])
    return train_df, eval_df, holdout_df, target_encoder

# REMOVING CORRELATED COLUMN (MEDIAN_LIST_PRICE) THE SECOND TIME
def removing_correlated_and_unwanted_columns(
        df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame,
        cols: list = ["median_list_price", "city", "city_full"]
        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df1.copy()
    eval_df = df2.copy()
    holdout_df = df3.copy()

    train_df = train_df.drop(columns=cols)
    eval_df = eval_df.drop(columns=cols)
    holdout_df = holdout_df.drop(columns=cols)
    return train_df, eval_df, holdout_df

# FULL FEATURE ENGINEERING PIPELINE
def feature_engineering_pipeline(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame):
    train = df1.copy()
    eval = df2.copy()
    holdout = df3.copy()
    # DATASETS TO BE USED FOR COMPUTING VIF AND MULTICOLLINEARITY
    train_df = engineer_date_features(train)
    eval_df = engineer_date_features(eval)
    holdout_df = engineer_date_features(holdout)
    # DATASETS WITHOUT COMPUTING VIF AND MULTICOLLINEARITY
    train_reserved = train.copy()
    eval_reserved = eval.copy()
    holdout_reserved = holdout.copy()
    # COLUMNS WITH HIGH VIF REMOVED FROM DATASETS
    OFFSET_COLUMNS = compute_variance_inflation_factor_optimized(train_df, target_col="price", vif_threshold=1000)
    train_df = train_df.drop(columns=OFFSET_COLUMNS)
    eval_df = eval_df.drop(columns=OFFSET_COLUMNS)
    holdout_df = holdout_df.drop(columns=OFFSET_COLUMNS)
    # REMOVE CORRELATED COLUMNS TO THE TARGET FROM BOTH DATASETS
    train_df, eval_df, holdout_df = removing_correlated_columns(train_df, eval_df, holdout_df)
    train_reserved, eval_reserved, holdout_reserved = removing_correlated_columns(train_reserved, eval_reserved, holdout_reserved)
    # ENGINEERING OTHER FEATURES WITH NUMERICAL AND CATEGORICAL COLUMNS
    """ZIPCODE FEATURE ENGINEERING"""
    train_df = engineering_zipcode_with_frequency_encoding(train_df)
    eval_df = engineering_zipcode_with_frequency_encoding(eval_df)
    holdout_df = engineering_zipcode_with_frequency_encoding(holdout_df)

    train_reserved = engineering_zipcode_with_frequency_encoding(train_reserved)
    eval_reserved = engineering_zipcode_with_frequency_encoding(eval_reserved)
    holdout_reserved = engineering_zipcode_with_frequency_encoding(holdout_reserved)
    """CITY_FULL FEATURE ENGINEERING"""
    train_df, eval_df, holdout_df, target_encoder_df = engineering_cityfull_with_target_encoder(train_df, eval_df, holdout_df)
    train_reserved, eval_reserved, holdout_reserved, target_encoder_reserved = engineering_cityfull_with_target_encoder(train_reserved, eval_reserved, holdout_reserved)
    # REMOVING THE COLUMN WITH HIGH VIF AFTER FEATURE ENGINEERING
    OFFSET_COLUMNS = compute_variance_inflation_factor_optimized(train_df, target_col="price", vif_threshold=5)
    train_df = train_df.drop(columns=OFFSET_COLUMNS)
    eval_df = eval_df.drop(columns=OFFSET_COLUMNS)
    holdout_df = holdout_df.drop(columns=OFFSET_COLUMNS)
    # REMOVING COLUMN CORRELATED WITH THE TARGET AFTER FEATURE ENGINEERING AND ALSO REMOVING ALL UNWANTED COLUMNS FROM BOTH DATASETS
    train_df, eval_df, holdout_df = removing_correlated_and_unwanted_columns(train_df, eval_df, holdout_df)
    train_reserved, eval_reserved, holdout_reserved = removing_correlated_and_unwanted_columns(train_reserved, eval_reserved, holdout_reserved)

    # SAVING ALL THE OUTPUTS FOR THE ENTIRE PIPELINE (TRAIN, EVAL, HOLDOUT) FOR FEATURE ENG WITH AND WITHOUT VIF
    # CREATING THE OUTPUT FOLDERS TO STORE THE OUTCOMES OF THIS STAGE
    output_path = os.path.join(os.getcwd(), "data", "feature_engineered")
    os.makedirs(output_path, exist_ok=True)
    model_path = os.path.join(os.getcwd(), "model", "feature_engineered_artifacts")
    os.makedirs(model_path, exist_ok=True)

    # SAVING THE OUTCOMES FILE INTO THE CREATED FOLDERS
    train_df.to_csv(os.path.join(output_path, "feat_eng_wtVIFtrained.csv"), index=False)
    eval_df.to_csv(os.path.join(output_path, "feat_eng_wtVIFeval.csv"), index=False)
    holdout_df.to_csv(os.path.join(output_path, "feat_eng_wtVIFholdout.csv"), index=False)

    train_reserved.to_csv(os.path.join(output_path, "feat_eng_wtoVIFtrained.csv"), index=False)
    eval_reserved.to_csv(os.path.join(output_path, "feat_eng_wtoVIFeval.csv"), index=False)
    holdout_reserved.to_csv(os.path.join(output_path, "feat_eng_wtoVIFholdout.csv"), index=False)

    # SAVING THE FEATURE ENGINEERING ENCODERS
    FEATURE = {
        "TARGETENCODER_WITH_VIF": target_encoder_df,
        "TARGETENCODER_WITHOUT_VIF": target_encoder_reserved
    }

    joblib.dump(
        FEATURE,
        os.path.join(model_path, "FEATURE.joblib"),
        compress=3
    )
    print(f"✅ Feature engineering artifacts saved to: {model_path}")
    print(f"✅ Fully feature engineered module completed and two datasets saved (withVIF and withoutVIF).")

if __name__ == "__main__":
    feature_engineering_pipeline(train, eval_df, holdout)








    

