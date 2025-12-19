import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# FIX: clean old state
os.environ.pop("MLFLOW_RUN_ID", None)
mlflow.end_run()

# ===============================
# MLflow tracking (CI/CD-safe)
# ===============================
mlflow.set_tracking_uri("sqlite:///mlflow.db")  # RELATIF, aman utk CI
mlflow.set_experiment("Sales Transaction - Linear Regression")

# Pastikan folder artifacts tersedia
os.makedirs("mlruns", exist_ok=True)

# ===============================
# Load dataset
# ===============================
df = pd.read_csv("Sales-Transaction-v.4a_preprocessing.csv")

y = df["Price"].astype(float)
X = df.drop(columns=["Price"]).astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    mlflow.sklearn.log_model(model, artifact_path="model")
