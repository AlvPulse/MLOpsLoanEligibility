from sklearn.base import BaseEstimator, TransformerMixin

class feature_eng_selection(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X=X.copy()
        X['monthly_payment'] = X['loan_amount'] / X['loan_term']
        X['income_after_loan'] = X['income_annum'] / 12 - X['monthly_payment']
        X['dependents_ratio'] = X['income_after_loan'] / (X['no_of_dependents'] + 1)
        X['total_assets'] = X[['commercial_assets_value', 'luxury_assets_value', 'bank_asset_value']].sum(axis=1)
        X.drop(['commercial_assets_value', 'luxury_assets_value','bank_asset_value','loan_id','monthly_payment'], axis=1, inplace=True)
        return X
