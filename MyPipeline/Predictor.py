import joblib
import pandas as pd
import numpy as np

class Predictor:
    def __init__(self, model_path: str):
        self.pipeline = joblib.load(model_path)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        df.columns = df.columns.str.strip()
        return np.where(self.pipeline.predict(df)==1,'Y','N')