import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def split_data(df, target: str, test_size=0.2, random_state=42):
    X = df.drop(target, axis=1)
    y = df[target]
    return train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)