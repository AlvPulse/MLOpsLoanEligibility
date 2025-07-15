import joblib
import pandas as pd

class Predictor:
    def __init__(self, model_path: str):
        self.pipeline = joblib.load(model_path)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        df.columns = df.columns.str.strip()
        return self.pipeline.predict(df)