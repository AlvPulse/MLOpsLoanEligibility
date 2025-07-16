import pathlib
import os

DATAPATH = "MyPipeline/DataSet"
FileName= 'loan_approval_dataset.csv'

MODEL_NAME = 'classification.pkl'
SAVE_MODEL_PATH = 'MyPipeline/trained_models'


Skew_handling_meth = 'yeo-johnson'
skew_handling_threshold = 0.55

CAT_COLS= ['education','self_employed']
NUM_COLS= ['income_annum', 'loan_amount', 'loan_term', 'cibil_score','income_after_loan','dependents_ratio','total_assets','no_of_dependents']
SKEW_COLS= ['residential_assets_value', 'income_after_loan', 'dependents_ratio']
