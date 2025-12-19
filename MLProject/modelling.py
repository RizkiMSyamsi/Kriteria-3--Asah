import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Hapus state lama (CI-safe)
os.environ.pop("MLFLOW_RUN_ID", None)
mlflow.end_run()

# Argumen dari MLProject
parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, required=True)
args = parser.parse_args()

# MLflow tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Sales Transaction - Linear Regression")

os.makedirs("mlruns", exist_ok=True)

# Load dataset
df = pd.read_csv(args.input_path)

y = df["Price"].astype(float)
X = df.drop(columns=["Price"]).astype(float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run(run_name="Linear Regression - Sales Price"):

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE:", mse)
    print("R2 :", r2)

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    mlflow.log_param("input_path", args.input_path)

    mlflow.sklearn.log_model(model, artifact_path="model")
