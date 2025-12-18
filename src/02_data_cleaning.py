### ============================ STAGE 02 - DATA CLEANING AND TRANSFORMATION MODULE +++++++++++++++++++++++++++++++++++++ ###
"""
FUNCTIONS TO CLEAN AND TRANSFORM DATASETS
- Loads train/eval/holdout datasets from ../data/raw
- Clean and normalizes city names.
- Maps cities to metros and merge with Lat/Long.
- Drops duplicates and extreme outliers.
- Saves cleaned datasets to ..data/cleaned
"""

