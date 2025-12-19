import pandas as pd
import mlflow
import mlflow.sklearn
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ============================
# MLFLOW LOCAL CONFIG
# ============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MLFLOW_DB_PATH = os.path.join(BASE_DIR, "mlflow.db")
ARTIFACT_ROOT = os.path.join(BASE_DIR, "mlruns")

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
mlflow.set_experiment("Sales Transaction - Linear Regression")

# ============================
# LOAD PREPROCESSED DATA
# ============================

DATA_PATH = os.path.join(BASE_DIR, "Sales-Transaction-v.4a_preprocessing.csv")
df = pd.read_csv(DATA_PATH)

# ============================
# FEATURE & TARGET
# ============================

y = df["Price"]
X = df.drop(columns=["Price"]).astype("float64")
y = y.astype("float64")

# ============================
# SPLIT
# ============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================
# ENABLE AUTOLOG
# ============================

mlflow.sklearn.autolog()

# ============================
# TRAIN
# ============================

with mlflow.start_run(run_name="Linear Regression - Sales Price"):

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Eval
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("MSE:", mse)
    print("R2 :", r2)

    # ============================
    # LOG MODEL (wajib untuk Docker)
    # ============================
    mlflow.sklearn.log_model(model, artifact_path="model")
