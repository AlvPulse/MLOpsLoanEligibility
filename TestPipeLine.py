import mlflow
from MyPipeline.Predictor import Predictor_MLflow, generate_fake_loan_data
import random
import numpy as np

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Generate fake data
    mlflow.set_tracking_uri("mysql+pymysql://root:P%40ssw0rd@localhost:3306/db_mlflow")
    model_name = "Predictor_Model_RF"
    model_stage = "Staging"
    model_uri = f"models:/{model_name}/{model_stage}"

    predictor = Predictor_MLflow(model_uri)
    Test_data = generate_fake_loan_data(10)
    Test_data['prediction'] = predictor.predict(Test_data)
    output_file = 'RandomTestDataResult.csv'
    Test_data.to_csv(output_file, index=False)
    print(f"Fake loan data with predictions generated and saved to '{output_file}'")




