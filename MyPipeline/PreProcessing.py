from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer


class SkewHandler(BaseEstimator, TransformerMixin):
    def __init__(self, SkewThreshold=0.55,method='yeo-johnson'):
        self.skew_threshold = SkewThreshold
        self.method= method
        self.to_transform = []
        self.transformer = PowerTransformer(method=method)

    def fit(self, X, y=None):
        skews= X.skew().abs()
        self.to_transform = skews[skews >= self.skew_threshold].index
        if self.to_transform:
            self.transformer.fit(X[self.to_transform])
    def transform(self, X, y=None):
        X = X.copy()
        if self.to_transform:
            X[self.to_transform] = self.transformer.transform(X[self.to_transform])
        return X

def get_preprocessor(numeric_cols, categorical_cols,Skew_threshold,Skew_handling_meth):
    numeric_transformer = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('skew', SkewHandler(SkewThreshold=Skew_threshold,method=Skew_handling_meth)),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('impute', SimpleImputer(strategy='mode')),
        ('LE', LabelEncoder()),
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
