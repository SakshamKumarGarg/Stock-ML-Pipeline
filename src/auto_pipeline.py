import subprocess
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = ROOT_DIR / "processed_data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


from yaml import safe_load
with open("config.yaml", "r", encoding="utf-8") as f:
    config = safe_load(f)



SYMBOLS = config.get("symbols", ["AAPL", "GOOGL", "TSLA"])

MODEL_TYPE = config.get("ml_training", {}).get("model_type", "RandomForest")



def run_phase(script_name):
    """Running a Python script from src """
    print(f"[INFO] Running {script_name}...")
    subprocess.run(["python", f"src/{script_name}"], check=True)

def generate_report(symbol):
    import numpy as np

    csv_files = sorted(PROCESSED_DIR.glob(f"{symbol}_processed_*.csv"))
    if not csv_files:
        return

    df = pd.read_csv(csv_files[-1], index_col=0, parse_dates=True)
    df.sort_index(inplace=True)

    df["Target"] = df["4. close"].pct_change().shift(-1)
    df.dropna(inplace=True)

    X = df.drop(columns=["Target"])
    y = df["Target"]

    model_files = list(MODELS_DIR.glob(f"{symbol}_*_model.pkl"))

    results = {}

    for file in model_files:
        name = file.name.split("_")[1]
        model = joblib.load(file)

        try:
            y_pred = model.predict(X)
        except:
            continue

        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        mse = np.mean((y - y_pred) ** 2)
        direction_acc = (np.sign(y_pred) == np.sign(y)).mean() * 100  # better for returns

        results[name] = {"MSE": mse, "R2": r2, "ACC": direction_acc}

    # ✅ Baseline model
    y_naive = df["4. close"].pct_change()
    y_naive = y_naive.shift(1)  # previous return as prediction

    # Align with y
    y_naive = y_naive.loc[y.index]

    # 🚨 Remove NaN rows (CRITICAL)
    valid_idx = y_naive.notna() & y.notna()

    y_naive = y_naive[valid_idx]
    y_valid = y[valid_idx]

    mse_naive = mean_squared_error(y_valid, y_naive)

    results["Naive"] = {"MSE": mse_naive, "R2": 0, "ACC": 0}

    # Save report
    report = REPORTS_DIR / f"{symbol}_report.txt"
    with open(report, "w") as f:
        f.write(f"{symbol} Report\n\n")

        for m, val in results.items():
            f.write(f"{m}\n")
            f.write(f"MSE: {val['MSE']:.4f}\n")
            f.write(f"R2: {val['R2']:.4f}\n")
            f.write(f"Accuracy: {val['ACC']:.2f}%\n")
            f.write("-"*30 + "\n")

    print(f"Report saved → {report}")


if __name__ == "__main__":
    # Phase 1: Fetching data
    run_phase("data_fetch.py")

    # Phase 2: Preprocessing
    run_phase("preprocess.py")

    # Phase 3: Training models
    run_phase("train_model.py")


    # Generate reports per symbol to analyze how stocks behaves
    for symbol in SYMBOLS:
        generate_report(symbol)
    print("Pipeline completed successfully!")
