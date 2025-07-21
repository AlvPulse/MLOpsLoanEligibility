import time

from MyPipeline.LoanPipeline import Build_training_pipeline
from MyPipeline.utils import load_data
from MyPipeline.Config import Skew_handling_meth, skew_handling_threshold, DATAPATH,FileName,SAVE_MODEL_PATH, MODEL_NAME
import joblib
from sklearn.model_selection import cross_val_score, KFold, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score
import numpy as np
from sklearn import set_config
import mlflow


def train_pipeline(model):
    file_path = DATAPATH + "/" + FileName
    df= load_data(file_path)
    model_path= SAVE_MODEL_PATH+"/"+ MODEL_NAME
    X= df.drop('loan_status', axis=1)
    y= np.where (df['loan_status']==' Approved', 1, 0)
    #print(y.sum())
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=41)
    set_config(transform_output='pandas')
    pipe= Build_training_pipeline(model,skew_handling_threshold,Skew_handling_meth)
    pipe.fit(X_train,y_train)
    train_score = pipe.score(X_train, y_train)
    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')  # or other scoring metrics
    joblib.dump(pipe,model_path)
    y_pred = pipe.predict(X_test)
    CV_score = cross_val_score(pipe, X, y, cv=5)
    #y_proba = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
        "train_score": train_score,
        "cv_score": CV_score,
        #"roc_auc": roc_auc_score(y_test, y_proba) if len(np.unique(y)) == 2 else None
        # ROC-AUC for binary classification
    }
    return pipe,metrics

def train_pipeline_mlflow(search_space: dict):
    file_path = DATAPATH + "/" + FileName
    df = load_data(file_path)
    model_path = SAVE_MODEL_PATH + "/" + MODEL_NAME
    X = df.drop('loan_status', axis=1)
    y = np.where(df['loan_status'] == ' Approved', 1, 0)
    # print(y.sum())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)
    set_config(transform_output='pandas')
    model_results=[]

    for model_name, config in search_space.items():
        mlflow.set_experiment(model_name)
        model=config["model"]
        param_grid = config["params"]
        pipe = Build_training_pipeline(model, skew_handling_threshold, Skew_handling_meth)
        search= RandomizedSearchCV(pipe, param_grid, cv=5, scoring='accuracy',n_iter=10)
        start_time = time.time()
        search.fit(X_train, y_train)
        end_time = time.time()

        best_params = search.best_params_
        best_score = search.best_score_
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        #y_proba = best_model.predict_proba(X_test)
        metrics = metric_finder(y_test,y_pred,end_time-start_time)
        with mlflow.start_run():
            mlflow.log_param("model", model_name)
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_model, "BestModel")
            model_results.append((model_name, best_model, metrics))

    sorted_model= sorted(model_results, key=lambda x: x[2]["accuracy"],reverse=True)
    top_model_name, top_model, top_metrics = sorted_model[0]
    best_model_path = f"{SAVE_MODEL_PATH}/{MODEL_NAME}_{top_model_name}.pkl"
    joblib.dump(top_model, best_model_path)

    print(f"\n✅ Best model: {top_model_name}")
    print(f"Metrics: {top_metrics}")

    return top_model, top_metrics


def metric_finder(y_test,y_pred, duration):
    metrics_inside = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='binary'),
        "recall": recall_score(y_test, y_pred, average='binary'),
        "f1_score": f1_score(y_test, y_pred, average='binary'),
        #"roc_auc": roc_auc_score(y_test, y_proba[:,1]),
        "train_time_sec": duration,
        #"train_score": best_model.score(X_train, y_train),
    }
    return metrics_inside

