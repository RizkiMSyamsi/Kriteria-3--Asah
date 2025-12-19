import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# =====================================================
# Hentikan run lama jika GitHub Actions mengatur MLFLOW_RUN_ID
# =====================================================
os.environ.pop("MLFLOW_RUN_ID", None)
mlflow.end_run()


# =====================================================
# MLFLOW CONFIG
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MLFLOW_DB_PATH = os.path.join(BASE_DIR, "mlflow.db")

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
mlflow.set_experiment("Sales Transaction - Linear Regression")


# =====================================================
# LOAD DATA
# =====================================================
DATA_PATH = os.path.join(BASE_DIR, "Sales-Transaction-v.4a_preprocessing.csv")
df = pd.read_csv(DATA_PATH)

# Features & Target
y = df["Price"].astype(float)
X = df.drop(columns=["Price"]).astype(float)


# =====================================================
# SPLIT DATA
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =====================================================
# ENABLE AUTOLOG
# =====================================================
mlflow.sklearn.autolog()


# =====================================================
# TRAINING & LOGGING
# =====================================================
with mlflow.start_run(run_name="Linear Regression - Sales Price"):
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE:", mse)
    print("R2 :", r2)

    # Log model 
    mlflow.sklearn.log_model(model, artifact_path="model")
