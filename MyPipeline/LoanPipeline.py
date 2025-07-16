from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
#from MyPipeline.PreProcessing import AutoColumnPreprocessor
from MyPipeline.Preprocess_static import AutoColumnPreprocessor
from MyPipeline.FetureEng import feature_eng_selection
from sklearn import set_config


def Build_training_pipeline(model,SkewThresh,Skew_handling_meth):
    #preproc= AutoColumnPreprocessor(SkewThresh,Skew_handling_meth)

    myPipeline = Pipeline([
        ('FeatureEngSelection', feature_eng_selection()),
        ('preprocessor', AutoColumnPreprocessor(skew_threshold=SkewThresh,Skew_handling_meth=Skew_handling_meth)),
        ('model', model)
    ])
    return myPipeline
