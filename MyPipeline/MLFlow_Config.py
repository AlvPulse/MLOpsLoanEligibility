from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

search_space = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000),
        "params": {
            "model__C": [0.1, 1.0, 10.0],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs"]
        }
    },
    "RandomForest": {
        "model": RandomForestClassifier(),
        "params": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [5, 10,None],
            "model__min_samples_split": [2, 5]
        }
    },
    "SVC": {
        "model": SVC(),
        "params": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["rbf", "linear"]
        }
    }
}