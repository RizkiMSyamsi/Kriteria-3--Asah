import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# =====================================================
# PATH & BASIC CONFIG
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "Sales-Transaction-v.4a_preprocessing.csv"
)

EXPERIMENT_NAME = "Sales Transaction - Linear Regression"


# =====================================================
# MLFLOW CONFIG
# =====================================================
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog(log_models=True)

print("MLflow Tracking URI :", mlflow.get_tracking_uri())
print("MLflow Experiment  :", EXPERIMENT_NAME)


# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully")
print("Shape   :", df.shape)
print("Columns :", df.columns.tolist())


# =====================================================
# FEATURE & TARGET
# =====================================================
TARGET_COL = "Price"

X = df.drop(columns=[TARGET_COL]).astype("float64")
y = df[TARGET_COL].astype("float64")


# =====================================================
# TRAIN TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape :", X_test.shape)


# =====================================================
# MODEL TRAINING
# =====================================================
model = LinearRegression()
model.fit(X_train, y_train)


# =====================================================
# EVALUATION
# =====================================================
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nTraining completed successfully")
print("MSE :", mse)
print("R2  :", r2)
