import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import yaml

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_TYPE = config.get("ml_training",{}).get("model_type","RandomForest")
TEST_SIZE = config.get("ml_training", {}).get("test_size", 0.2)
RANDOM_STATE = config.get("ml_training", {}).get("random_state", 42)

processed_directory = Path("processed_data")
Models_directory = Path("models")
Models_directory.mkdir(exist_ok=True)

csv_files = sorted(processed_directory.glob("*_processed_*.csv"))
if not csv_files:
    raise FileNotFoundError("No processed CSVs found. Run Phase 2 first.")

for csv_path in csv_files:
    try:
        symbol = csv_path.name.split("_")[0]
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        print(f"[INFO] Loaded {symbol} dataset: {df.shape}")

        # Target: next day's close
        df.sort_index(inplace=True)
        df["Target"] = df["4. close"].shift(-1)
        df.dropna(inplace=True)

        X = df.drop(columns=["Target"])
        y = df["Target"]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
        )

        # Initialize model
        if MODEL_TYPE == "RandomForest":
            model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        elif MODEL_TYPE == "LinearRegression":
            model = LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {MODEL_TYPE}")

        # Train
        print(f"[INFO] Training {MODEL_TYPE} model for {symbol}...")
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"{symbol} evaluation → MSE: {mse:.4f}, R²: {r2:.4f}")

        # Save model
        date_str = datetime.now().strftime("%Y-%m-%d")
        model_file = Models_directory / f"{symbol}_{MODEL_TYPE}_model_{date_str}.pkl"
        joblib.dump(model, model_file)
        print(f"{symbol} model saved → {model_file}\n")

    except Exception as e:
        print(f"[ERROR] Failed to train model for {symbol}: {e}")