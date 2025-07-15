from sklearn.pipeline import Pipeline
from PreProcessing import get_preprocessor
from FetureEng import feature_eng_selection

def Build_training_pipeline(model,numerical_cols,categorical_cols,SkewThresh,Skew_handling_meth):
    preproc= get_preprocessor(numerical_cols,categorical_cols,SkewThresh,Skew_handling_meth)
    myPipeline = Pipeline([
        ('FeatureEngSelection', feature_eng_selection()),
        ('preprocessor', preproc),
        ('model', model)
    ])
    return myPipeline
