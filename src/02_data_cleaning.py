### ============================ STAGE 02 - DATA CLEANING AND TRANSFORMATION MODULE =================================== ###
"""
FUNCTIONS TO CLEAN AND TRANSFORM DATASETS
- Loads train/eval/holdout datasets from ../data/raw
- Loads the metro_dataset to map the cities with latitude and longitude from the metro data.
- Clean and normalizes city names.
- Maps cities to metros and merge with Lat/Long.
- Drops duplicates and extreme outliers.
- Saves cleaned datasets to ..data/cleaned.
"""

# IMPORT NECESSARY LIBRARIES
import numpy as np
import pandas as pd
import os
import re
import config
import difflib
import matplotlib.pyplot as plt
import seaborn as sb

# PATH TO THE METRO DATA
metro_data_and_path = config.METRO_DATA_PATH["METRO_PATH"]

# LOADS ALL THE DATA
metro = pd.read_csv(metro_data_and_path)
train = pd.read_csv("./data/raw/train.csv")
eval_df = pd.read_csv("./data/raw/eval.csv")
holdout = pd.read_csv("./data/raw/holdout.csv")

# MERGING THE CITIES IN THE TRAINING DATA TO THE LAT/LONG IN THE METRO DATA
"""
Normalize city names before comparison
"""
def city_normalizer(city: str) -> str:
    if pd.isna(city):
        return city
    city = str(city)
    city = city.lower()
    city = city.replace("_", "-")
    city = re.sub(r"[^\w\s\-]", "", city)
    city = re.sub(r"\s+", "-", city).strip()
    return city

"""
Extract the cities in training and also the cities in metro to check those that aligned.
"""
def cities_check(train: pd.DataFrame = train, metro: pd.DataFrame = metro, train_col: str = "city_full", metro_col: str = "metro_full") -> dict[str, str]:
    """
    The function extract the city names in training and metro data then check for city in training but not in metro data and use difflib to match cities in both data.
    Arguments:
    - train df: train default
    - metro df: metro default
    - columns: city_full and metro_full default
    Returns:
    - Dict: Dictionary containing matching cities ONLY.
    """

    train_cities = train[train_col].unique().tolist()
    metro_cities = metro[metro_col].unique().tolist()

    """
    Checking for cities in train data but not in metro data.
    """
    """missing = [item for item in train_cities if item not in metro_cities]"""

    """
    For matching of the cities, the difflib module which finds approximate matches for strings and sometimes called fuzzy.
    """
    MAPPING = {
        item: difflib.get_close_matches(item, metro_cities, n=1, cutoff=0.6)[0]
        if difflib.get_close_matches(item, metro_cities, n=1, cutoff=0.6)
        else "Washington-Arlington-Alexandria"
        for item in train_cities if item not in metro_cities
    }
    """Applying the city_normalizer function to normalize the city names (string)"""
    MAPPING = {city_normalizer(key): city_normalizer(value) for key, value in MAPPING.items()}
    return MAPPING


"""
Function to correct the city names mismatched, and merge corrected city with lat/long.
"""
def city_mapper_with_lat_and_long(df: pd.DataFrame, metro: pd.DataFrame = metro, train_col: str = "city_full", metro_col: str = "metro_full") -> pd.DataFrame:

    train_cities = df[train_col].unique().tolist()
    metro_cities = metro[metro_col].unique().tolist()

    MAPPING = {
        item: difflib.get_close_matches(item, metro_cities, n=1, cutoff=0.6)[0]
        if difflib.get_close_matches(item, metro_cities, n=1, cutoff=0.6)
        else "Washington-Arlington-Alexandria"
        for item in train_cities if item not in metro_cities
    }
    """Applying the city_normalizer function to normalize the city names (string)"""
    MAPPING = {city_normalizer(key): city_normalizer(value) for key, value in MAPPING.items()}

    # df["city_full_norm"] = df[train_col].apply(city_normalizer)
    # df["city_full_norm"] = df["city_full_norm"].replace(MAPPING)
    df[train_col] = df[train_col].apply(city_normalizer)
    df[train_col] = df[train_col].replace(MAPPING)

    # metro["metro_full_norm"] = metro["metro_full"].apply(city_normalizer)
    metro["metro_full"] = metro["metro_full"].apply(city_normalizer)

    df = df.merge(
        metro[["metro_full", "lat", "lng"]],
        how="left",
        left_on=train_col,
        right_on="metro_full"
    )

    df.drop(columns=["metro_full"], inplace=True)
    
    """LOGGING ANY CITIES WITHOUT MATCH"""
    NO_MATCH = df[df["lat"].isnull()][train_col].unique()
    if len(NO_MATCH) > 0:
        print(f"❌ The Latitude and Longitude still has some no_match cities after normalization and mapping: {list(NO_MATCH)}")
    else:
        print("✅ All cities in train data matched with cities in the metro dataset.")
    
    return df

# REMOVING THE DUPLICATED ROWS
def removing_duplicates(df: pd.DataFrame, cols: list = ["date", "year"]) -> pd.DataFrame:
    """
    Removing the duplicated rows excluding dates/years.
    """
    before = df.shape[0]
    df = df.drop_duplicates(subset=df.columns.difference(cols), keep=False)
    df.reset_index(drop=True)
    after = df.shape[0]
    print(f"✅ Removed {before - after} duplicated rows excluding date/year.")
    return df

# REMOVING THE OUTLIERS
"""
I benchmarked with the median_list_price and removed the outlier above the threshold
"""
def removing_outliers(df: pd.DataFrame, col_outlier: str = "median_list_price", threshold: float = 19000000) -> pd.DataFrame:
    before = df.shape[0]
    df = df[df[col_outlier] <= threshold].copy()
    after = df.shape[0]
    print(f"✅ Removed {before - after} outliered rows using the median_list_price.")
    return df

# FULL DATA CLEANING PIPELINE
"""
COMPLETE DATA CLEANING PIPELINE
"""
def full_cleaning_pipeline(df:pd.DataFrame, metro: pd.DataFrame) -> pd.DataFrame:
    df = city_mapper_with_lat_and_long(df, metro)
    df = removing_duplicates(df)
    df = removing_outliers(df)
    return df

# SAVING THE OUTCOMES OF THIS STAGE
def saving_output_files(train: pd.DataFrame, eval_: pd.DataFrame, holdout: pd.DataFrame, metro: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = full_cleaning_pipeline(train, metro)
    eval_df = full_cleaning_pipeline(eval_, metro)
    holdout_df = full_cleaning_pipeline(holdout, metro)

    # CREATING THE OUTPUT FOLDERS TO STORE THE OUTCOMES OF THIS STAGE
    output_path = os.path.join(os.getcwd(), "data", "cleaned")
    os.makedirs(output_path, exist_ok=True)

    # SAVING THE OUTCOMES FILE INTO THE CREATED FOLDERS
    train_df.to_csv(os.path.join(output_path, "train_cleaned.csv"), index=False)
    eval_df.to_csv(os.path.join(output_path, "eval_cleaned.csv"), index=False)
    holdout_df.to_csv(os.path.join(output_path, "holdout_cleaned.csv"), index=False)

    print(f"✅ Splited data loaded from raw, city mapped with lat/long, duplicated and outlier rows removed and saved cleaned data to {output_path}.")
    print(f"  Training: {train_df.shape}, Evaluation: {eval_df.shape}, Holdout: {holdout_df.shape}")

    return train_df, eval_df, holdout_df
    

if __name__ == "__main__":
    print("Running data cleaner module...")
    train_df, eval_df, holdout_df = saving_output_files(train, eval_df, holdout, metro)
