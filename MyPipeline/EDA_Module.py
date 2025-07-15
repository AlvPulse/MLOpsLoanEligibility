import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
def plot_dataset(df):
    n_cols = 3  # Number of plots per row

    # Select all columns
    columns = df.columns
    n = len(columns)

    # Create subplots
    fig, axes = plt.subplots(nrows=-(-n // n_cols), ncols=n_cols,
                             figsize=(5 * n_cols, 4 * ((n + n_cols - 1) // n_cols)))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        ax = axes[i]
        if pd.api.types.is_numeric_dtype(df[col]):
            sns.histplot(df[col].dropna(), ax=ax, kde=False, bins=20)
        else:
            df[col].value_counts().plot.pie(ax=ax, autopct='%1.1f%%', startangle=90)
            ax.set_ylabel('')
        ax.set_title(col)

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()





def transform_skewed_features(df, cols, method='log',bin_count=5):
    df_transformed = df.copy()

    for col in cols:
        if method == 'log':
            df_transformed[col] = np.log1p(df[col])

        elif method == 'bin':
            df_transformed[col] = pd.qcut(df[col], q=bin_count, labels=False, duplicates='drop')

        elif method == 'power':
            pt = PowerTransformer(method='yeo-johnson')
            df_transformed[col] = pt.fit_transform(df[[col]])

        elif method == 'rank':
            df_transformed[col] = df[col].rank(method='average')

        elif method == 'winsor':
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df_transformed[col] = df[col].clip(lower, upper)

        else:
            raise ValueError(f"Unknown method: {method}")

    return df_transformed
