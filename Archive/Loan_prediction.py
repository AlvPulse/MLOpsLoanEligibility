import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from MyPipeline.EDA_Module import transform_skewed_features

def preprocess_dataset(file_name="loan_approval_dataset.csv",skew_handling_method='bin'):
    df= pd.read_csv(file_name)

    categorical_columns = [' education', ' self_employed', ' loan_status']
    discrete_columns = categorical_columns + ['loan_id', ' no_of_dependents']


    probably_skewed_columns = [' commercial_assets_value', ' luxury_assets_value', ' bank_asset_value',
                               ' residential_assets_value', ' loan_amount']

    idx_skewed = abs(df[probably_skewed_columns].skew()) > 0.55
    skewed_columns = idx_skewed.index
    if skew_handling_method == 'bin':
        discrete_columns += skewed_columns.tolist()

    df_normalized= transform_skewed_features(df, skewed_columns,skew_handling_method)

    df_normalized['monthly_payment']= df_normalized[' loan_amount']/df_normalized[' loan_term']
    df_normalized['Monthly_income_after_loan']= df_normalized[' income_annum']/12 - df_normalized['monthly_payment']
    df_normalized['dependent_remaining']=  df_normalized['Monthly_income_after_loan']/ (df_normalized[' no_of_dependents']+1)

    df_normalized['Total_assets']= df_normalized[[' commercial_assets_value', ' luxury_assets_value',' bank_asset_value']].sum(axis=1)
    #df_normalized['Assets_to_loan_ratio']= df_normalized['Total_assets']/df_normalized[' loan_amount']

    for column in categorical_columns:
        df_normalized[column] = LabelEncoder().fit_transform(df_normalized[column])
    total_columns=list(df_normalized.columns)
    for cat_column in discrete_columns:
        total_columns.remove(cat_column)
    numerical_columns =total_columns
    df_scaled= df_normalized.drop('loan_id',axis=1).copy()
    df_scaled[numerical_columns]= StandardScaler().fit_transform(df_scaled[numerical_columns])
    X_train, X_test, y_train, y_test = train_test_split(df_scaled.drop(' loan_status',axis=1), df_scaled[' loan_status'] , test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test