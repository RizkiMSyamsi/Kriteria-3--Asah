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
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(BASE_DIR,'mlflow.db')}")
mlflow.set_experiment("Sales Transaction - Linear Regression")

# LOAD DATA
df = pd.read_csv(os.path.join(BASE_DIR, "Sales-Transaction-v.4a_preprocessing.csv"))

y = df["Price"].astype(float)
X = df.drop(columns=["Price"]).astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DISABLE AUTOLOG (prevent Windows path injection)
# mlflow.sklearn.autolog()

with mlflow.start_run(run_name="Linear Regression - Sales Price"):

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE:", mse)
    print("R2 :", r2)

    # MANUAL LOGGING (aman untuk CI/CD)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(model, "model")
