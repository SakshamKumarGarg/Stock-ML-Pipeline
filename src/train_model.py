import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib
import yaml

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

# Load config
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# Paths
processed_directory = Path("processed_data")
Models_directory = Path("models")
Models_directory.mkdir(exist_ok=True)

csv_files = sorted(processed_directory.glob("*_processed_*.csv"))

# Models
models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(max_depth=5),
    "RandomForest": RandomForestRegressor(n_estimators=100),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=100),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100),
    "XGBoost": XGBRegressor(n_estimators=100),
}

for csv_path in csv_files:
    symbol = csv_path.name.split("_")[0]
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

    df.sort_index(inplace=True)
    df["Target"] = df["4. close"].pct_change().shift(-1)
    df.replace([float("inf"), -float("inf")], None, inplace=True)
    df.dropna(inplace=True)

    X = df.drop(columns=["Target"])
    y = df["Target"]

    # ✅ TIME SERIES SPLIT (NO SHUFFLE)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\n[INFO] Training models for {symbol}")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"{symbol} | {name} → MSE: {mse:.4f}, R2: {r2:.4f}")

        file = Models_directory / f"{symbol}_{name}_model.pkl"
        joblib.dump(model, file)

print("✅ Training complete")