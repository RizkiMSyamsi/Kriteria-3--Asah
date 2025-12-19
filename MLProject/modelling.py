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

# SETUP TRACKING
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MLFLOW_DB = os.path.join(BASE_DIR, "mlflow.db")
ARTIFACT_DIR = os.path.join(BASE_DIR, "mlruns")

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB}")
mlflow.set_experiment("Sales Transaction - Linear Regression")

# FIX: FORCE artifact location
os.environ["MLFLOW_ARTIFACT_ROOT"] = ARTIFACT_DIR

# LOAD DATA
df = pd.read_csv(os.path.join(BASE_DIR, "Sales-Transaction-v.4a_preprocessing.csv"))

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

    # MANUAL LOGGING
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    
    mlflow.sklearn.log_model(model, name="model")
