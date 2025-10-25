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
with open(ROOT_DIR / "config.yaml") as f:
    config = safe_load(f)



SYMBOLS = config.get("symbols", ["AAPL", "GOOGL", "TSLA"])

MODEL_TYPE = config.get("ml_training", {}).get("model_type", "RandomForest")



def run_phase(script_name):
    """Running a Python script from src """
    print(f"[INFO] Running {script_name}...")
    subprocess.run(["python", f"src/{script_name}"], check=True)

def generate_report(symbol):
    """Generate a small sample report and plot per symbol"""
    # Load latest processed CSV that we call at last in program for every iteration
    csv_files = sorted(PROCESSED_DIR.glob(f"{symbol}_processed_*.csv"))
    if not csv_files:
        print(f"[WARN] No processed CSV for {symbol}")
        return
    df = pd.read_csv(csv_files[-1], index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    df["Target"] = df["4. close"].shift(-1)
    df.dropna(inplace=True)
    X = df.drop(columns=["Target"])
    y = df["Target"]

    #Load the model for every symbol
    model_files = sorted(MODELS_DIR.glob(f"{symbol}_{MODEL_TYPE}_model_*.pkl"))
    if not model_files:
        print(f"[WARN] No model for {symbol}")
        return
    model = joblib.load(model_files[-1])


    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

   
   # creating a report by analyzing MSE from above model 
    report_file = REPORTS_DIR / f"{symbol}_report_{datetime.now().strftime('%Y-%m-%d')}.txt"
    with open(report_file, "w") as f:
        f.write(f"{symbol} Report\n")
        f.write(f"MSE: {mse:.4f}\nR²: {r2:.4f}\n")
    print(f"Report saved → {report_file}")

   
   # here the plotting 
    plt.figure(figsize=(6,3))
    plt.plot(df.index, y, label="Actual Close")
    plt.plot(df.index, y_pred, label="Predicted Close")
    plt.title(f"{symbol} Actual vs Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / f"{symbol}_plot_{datetime.now().strftime('%Y-%m-%d')}.png")
    plt.close()



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
