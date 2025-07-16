from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer


class SkewHandlerTH(BaseEstimator, TransformerMixin):
    def __init__(self, skth=0.55, method='yeo-johnson'):
        self.skth = skth  # Changed to self.skth
        self.method = method
        self.to_transform = []
        self.transformer = PowerTransformer(method=method)

    def fit(self, X, y=None):
        skews = X.skew().abs()
        self.to_transform = skews[skews >= self.skth].index.tolist()
        #print(self.to_transform)
        if self.to_transform:
            self.transformer.fit(X[self.to_transform])
        return self  # Added return for scikit-learn compatibility

    def transform(self, X, y=None):
        X = X.copy()
        if self.to_transform:
            X[self.to_transform] = self.transformer.transform(X[self.to_transform])
        return X

class AutoColumnPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, skew_threshold=0.55,Skew_handling_meth='yeo-johnson'):
        self.skew_threshold = skew_threshold
        self.Skewmethod= Skew_handling_meth

    def fit(self, X, y=None):
        numeric_cols = X.select_dtypes(include='number').columns.tolist()
        #print(numeric_cols)
        Total_columns = X.columns.tolist()
        for col in numeric_cols:
            Total_columns.remove(col)
        self.numeric_cols = numeric_cols
        self.categorical_cols = Total_columns
        #print(self.categorical_cols)
        self.pipeline = ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('skew', SkewHandlerTH(self.skew_threshold,self.Skewmethod)),
                ('scale', StandardScaler())
            ]), self.numeric_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encode', OrdinalEncoder())
            ]), self.categorical_cols)
        ])

        self.pipeline.fit(X)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)

#
# def get_preprocessor(Skew_threshold,Skew_handling_meth):
#
#     numeric_transformer = Pipeline([
#         ('impute', SimpleImputer(strategy='median')),
#         ('skew', SkewHandler(SkewThreshold=Skew_threshold,method=Skew_handling_meth)),
#         ('scaler', StandardScaler())
#     ])
#     categorical_transformer = Pipeline([
#         ('impute', SimpleImputer(strategy='mode')),
#         ('LE', LabelEncoder()),
#     ])
#     preprocessor = ColumnTransformer([
#         ('num', numeric_transformer, numeric_cols),
#         ('cat', categorical_transformer, categorical_cols)
#     ])
#     return preprocessor
