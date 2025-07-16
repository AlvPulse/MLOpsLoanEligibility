from MyPipeline.LoanPipeline import Build_training_pipeline
from MyPipeline.utils import load_data
from MyPipeline.Config import Skew_handling_meth, skew_handling_threshold, DATAPATH,FileName,SAVE_MODEL_PATH, MODEL_NAME
import joblib
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,roc_auc_score
import numpy as np
from sklearn import set_config


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

