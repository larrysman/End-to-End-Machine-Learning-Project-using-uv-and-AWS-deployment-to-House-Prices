### ============ UNIT TEST FOR THE DATA LOADER SCRIPT - STAGE 01 ======== ###

import pandas as pd
import pytest
import os

from src import data_loader_01

# load_data = data_loader_01.load_data
# data_sample = data_loader_01.data_sample
# convert_date_column = data_loader_01.convert_date_column
# splitting_data = data_loader_01.splitting_data
# saving_output_files = data_loader_01.saving_output_files

# =====================================
# LOAD DATA - unit test
# =====================================

def test_load_data(tmp_path):
    dummy_path = tmp_path/'raw.csv'

    df = pd.DataFrame({
        'date': pd.date_range('2018-01-01', periods=6, freq='365D'),
        'price': [100, 200, 300, 400, 500, 600],
        'zipcode': [1000, 2000, 1000, 2000, 3000, 4000],
        'city_full': ['A', 'B', 'A', 'B', 'C', 'D']
    })

    df.to_csv(dummy_path, index=False)

    df_loaded = data_loader_01.load_data(data_path=str(dummy_path))
    # df_loaded = load_data(data_path=str(dummy_path))

    # ==================================
    # ASSERTION FOR DATA QUALITY CHECKER
    # ==================================

    # ASSERT THAT DATAFRAME EXISTS - LOADED DATAFRAME CHECKER
    assert df_loaded is not None

    # ASSERT THAT NUMBER OF ROWS IS CORRECT - ROW COUNT MISMATCH CHECKER
    assert len(df_loaded) == 6

    # ASSERT THAT COLUMNS NAMES EXIST - COLUMNS NAMES MISMATCH CHECKER
    expected_cols = ['date', 'price', 'zipcode', 'city_full']
    assert list(df_loaded.columns) == expected_cols

    # ASSERT THAT PRICE DATATYPE IS NUMERIC - PRICE COLUMN DATATYPE CHECKER
    assert pd.api.types.is_numeric_dtype(df_loaded['price'])

    # ASSERT SPECIFIC VALUE TO ENSURE CORRECT PARSING - FIRST PRICE VALUE CHECKER
    assert df_loaded.loc[0, 'price'] == 100
    print("✅ Data Loader Test Passed!")


# =================================
# TAKE SAMPLE DATA Unit Test
# =================================

def test_data_sample():
    # CREATE A SIMPLE DATAFRAME
    df = pd.DataFrame({'value': range(10)})

    # SAMPLE FRACTION = NONE TO RETURN ORIGINAL DF
    result_none = data_loader_01.data_sample(df, sample_frac=None)
    # If no sampling fraction is provided, the function should not modify the data
    assert result_none.equals(df)

    # INVALID FRACTIONS (<=0 OR >=1) -> RETURN ORIGINAL DF
    result_zero = data_loader_01.data_sample(df, sample_frac=0)
    result_one = data_loader_01.data_sample(df, sample_frac=1)
    # The fractions <=0 or >=1 are not meaningful for sampling, hence the funtion correctly returns the original DataFrame
    assert result_zero.equals(df)
    assert result_one.equals(df)

    # VALID FRACTION -> SAMPLE CORRECTLY
    result_valid = data_loader_01.data_sample(df, sample_frac=0.5, random_state=42)
    # Assert correct number of rows - by 50% we expect exactly 5 rows
    assert len(result_valid) == 5
    # Assert samples values come from original df - ensures no new or unexpected values in the dataframe
    assert set(result_valid['value']).issubset(set(df['value']))
    # Assert index is reset - this ensures that the DataFrame is explicitly resets the index
    assert list(result_valid.index) == list(range(5))
    print("✅ Sampled Data Test Passed!")


# ===================================
# CONVERT DATE COLUMN UNIT TEST
# USING MONKEYPATCH: Monkeypatch is needed only when a function internally calls another function that you want to replace.
# ===================================

def test_convert_date_column(monkeypatch):
    # CREATE THE DUMMY UNSORTED DATA WITH STRING DATES
    df_input = pd.DataFrame({
        "date": ["2021-01-05", "2019-12-31", "2020-06-01", "2018-03-10", "2022-01-01"],
        "value": [10, 20, 30, 40, 50]
    })

    # MOCK load_data() TO RETURN OUR DUMMY DATAFRAME
    monkeypatch.setattr(
        data_loader_01, 
        "load_data", 
        lambda: df_input
    )

    # CALLING THE FUNCTION
    df_output = data_loader_01.convert_date_column(date_column="date")

    # =====================================
    # ASSERTIONS FOR DATE COLUMN CONVERSION
    # =====================================

    # DATE COLUMN MUST BE CONVERTED TO DATETIME
    assert pd.api.types.is_datetime64_any_dtype(df_output["date"])

    # DATA MUST BE SORTED IN ASCENDING DATE ORDER
    assert list(df_output["date"]) == sorted(df_output["date"])

    # INDEX MUST BE RESET AFTER SORTING
    assert list(df_output.index) == list(range(len(df_output)))

    # TAKING 60% SAMPLE MUST RDUCE THE NUMBER OF ROWS
    expected_length = int(len(df_input) * 0.6)
    assert len(df_output) == expected_length
    print("✅ Date Column Conversion Test Passed!")


# ==========================================
# DATA SPLITTING FOR TEMPORAL DATA UNIT TEST
# ==========================================

def test_splitting_data(monkeypatch):
    # CREATE DUMMY TEMPORAL DATA POSSIBLE FOR SPLITTING INTO TRAIN, TEST, HOLDOUT
    df_input = pd.DataFrame({
        "date": pd.to_datetime([
            "2018-01-01",   # train
            "2019-12-31",   # train boundary
            "2020-01-01",   # eval start
            "2021-12-31",   # eval end
            "2022-01-01",   # holdout start
            "2023-06-01"    # holdout
        ]),
        "value": [1, 2, 3, 4, 5, 6]
    })

    # MOCK THE convert_date_column() TO RETURN THE DUMMY DATAFRAME
    monkeypatch.setattr(
        data_loader_01,
        "convert_date_column",
        lambda: df_input
    )

    # CALLING THE FUNCTION
    train_df, eval_df, holdout_df = data_loader_01.splitting_data(date_column="date")

    # ==============================
    # ASSERTIONS FOR DATA SPLITTING
    # ==============================

    # TRAINING SET: dates <= 2019-12-31
    assert train_df["date"].max() <= pd.to_datetime("2019-12-31")
    assert len(train_df) == 2

    # EVALUATION SET: 2020-01-01 to 2021-12-31
    assert eval_df["date"].min() >= pd.to_datetime("2020-01-01")
    assert eval_df["date"].max() <= pd.to_datetime("2021-12-31")
    assert len(eval_df) == 2

    # HOLDOUT SET: dates >= 2022-01-01
    assert holdout_df["date"].min() >= pd.to_datetime("2022-01-01")
    assert len(holdout_df) == 2
    print("✅ Data Splitting Test Passed!")


# =========================================
# SAVING THE OUTPUTS UNIT TEST
# =========================================

def test_saving_output_files(tmp_path, monkeypatch):
    # CREATE DUMMY SPLIT DATAFRAMES
    train_df = pd.DataFrame({"date": ["2018-01-01"], "value": [1]})
    eval_df = pd.DataFrame({"date": ["2020-01-01"], "value": [2]})
    holdout_df = pd.DataFrame({"date": ["2022-01-01"], "value": [3]})

    # MOCK THE splitting_data() SO THAT IT RETURNS THE DUMMY DATAFRAMES
    monkeypatch.setattr(
        data_loader_01,
        "splitting_data",
        lambda: (train_df, eval_df, holdout_df)
    )

    # MOCK os.getcwd() SO FILES ARE SAVED INSIDE THE pytest's tmp_path
    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))

    # CALLING THE FUNCTION
    out_train, out_eval, out_holdout = data_loader_01.saving_output_files()

    # ============================
    # ASSERTIONS FOR SAVING FILES
    # ============================

    # RETURNED DATAFRAMES MUST MATCH THE MOCKED ONES
    assert out_train.equals(train_df)
    assert out_eval.equals(eval_df)
    assert out_holdout.equals(holdout_df)

    # CHECK THAT THE OUTPUT FOLDER WAS CREATED
    output_path = tmp_path / "data" / "raw"
    assert output_path.exists()

    # CHECK THAT THE CSV FILES WERE CREATED
    assert (output_path / "train.csv").exists()
    assert (output_path / "eval.csv").exists()
    assert (output_path / "holdout.csv").exists()

    # CHECK THAT THE SAVED CSV CONTENTS MATCH THE ORIGINAL DATAFRAMES
    saved_train = pd.read_csv(output_path / "train.csv")
    saved_eval = pd.read_csv(output_path / "eval.csv")
    saved_holdout = pd.read_csv(output_path / "holdout.csv")

    assert saved_train.equals(train_df)
    assert saved_eval.equals(eval_df)
    assert saved_holdout.equals(holdout_df)
    print("✅ Data Successfully Saved Test Passed!")