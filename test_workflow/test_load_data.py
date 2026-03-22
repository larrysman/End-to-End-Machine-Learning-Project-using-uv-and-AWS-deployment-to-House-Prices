### ============ UNIT TEST FOR THE DATA LOADER SCRIPT - STAGE 01 ======== ###

import pandas as pd
import pytest

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
    assert result_none.equals(df)

    # INVALID FRACTIONS (<=0 OR >=1) -> RETURN ORIGINAL DF
    result_zero = data_loader_01.data_sample(df, sample_frac=0)
    result_one = data_loader_01.data_sample(df, sample_frac=1)
    assert result_zero.equals(df)
    assert result_one.equals(df)

    # VALID FRACTION -> SAMPLE CORRECTLY
    result_valid = data_loader_01.data_sample(df, sample_frac=0.5, random_state=42)
    # Assert correct number of rows
    assert len(result_valid) == 5
    # Assert samples values come from original df
    assert set(result_valid['value']).issubset(set(df['value']))
    # Assert index is reset
    assert list(result_valid.index) == list(range(5))
    print("✅ Sampled Data Test Passed!")