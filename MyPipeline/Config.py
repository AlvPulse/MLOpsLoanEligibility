import pathlib
import os

DATAPATH = "MyPipeline/DataSet"
FileName= 'loan_approval_dataset.csv'

MODEL_NAME = 'classification.pkl'
SAVE_MODEL_PATH = 'MyPipeline/trained_models'


Skew_handling_meth = 'yeo-johnson'
skew_handling_threshold = 0.55

