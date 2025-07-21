import joblib
import pandas as pd
import numpy as np
import mlflow
import random

class Predictor_joblib:
    def __init__(self, model_path: str):
        self.pipeline = joblib.load(model_path)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        df.columns = df.columns.str.strip()
        return np.where(self.pipeline.predict(df)==1,'Y','N')

class Predictor_MLflow:
    def __init__(self, model_uri: str):
        self.pipeline = mlflow.pyfunc.load_model(model_uri)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        df.columns = df.columns.str.strip()
        return np.where(self.pipeline.predict(df)==1,'Y','N')


def generate_fake_loan_data(num_records=100):
    # Define possible values
    education_options = [' Graduate', ' Not Graduate']
    self_employed_options = [' Yes', ' No']


    # Generate data
    data = {
        'loan_id': range(1, num_records + 1),
        'no_of_dependents': [random.randint(0, 5) for _ in range(num_records)],
        'education': [random.choice(education_options) for _ in range(num_records)],
        'self_employed': [random.choice(self_employed_options) for _ in range(num_records)],
        'income_annum': [random.randint(1000000, 10000000) for _ in range(num_records)],
        'loan_amount': [random.randint(1000000, 30000000) for _ in range(num_records)],
        'loan_term': [random.randint(6, 36) for _ in range(num_records)],
        'cibil_score': [random.randint(300, 900) for _ in range(num_records)],
        'residential_assets_value': [random.randint(0, 25000000) for _ in range(num_records)],
        'commercial_assets_value': [random.randint(0, 20000000) for _ in range(num_records)],
        'luxury_assets_value': [random.randint(0, 30000000) for _ in range(num_records)],
        'bank_asset_value': [random.randint(0, 10000000) for _ in range(num_records)],
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    return df