### ====== ALLOWING MODULES TO BE IMPORTED ACROSS FILES BECAUSE OF SCRIPT NAMING NOMENCLATURE THAT START WITH 01...06 ==== ###

import importlib

# LOAD THE NUMERIC MODULES UNDER VALID PYTHON NAMES

data_loader_01 = importlib.import_module('.01_data_loader', package=__name__)
# data_cleaning_02 = importlib.import_module('.02_data_cleaning', package=__name__)
# feature_engineering_03 = importlib.import_module('.03_feature_engineering', package=__name__)
# preprocessing_with_vif_04 = importlib.import_module('.04_preprocessing_with_vif', package=__name__)
# model_training_05 = importlib.import_module('.05_model_training', package=__name__)
# hyperparameter_tuning_06 = importlib.import_module('.06_hyperparameter_tuning', package=__name__)
