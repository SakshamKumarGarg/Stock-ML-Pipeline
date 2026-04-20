import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import yaml


with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

TEST_SIZE = config.get("ml_training", {}).get("test_size", 0.2)
RANDOM_STATE = config.get("ml_training", {}).get("random_state", 42)

SYMBOLS = config.get("symbols", ["AAPL"])
MA_DAYS = config.get("preprocessing", {}).get("moving_average_days", [3, 7])
STD_DAYS = config.get("preprocessing", {}).get("rolling_std_days", [3, 7])

data_directory = Path("data")
PROCESSED_DIR = Path("processed_data")
PROCESSED_DIR.mkdir(exist_ok=True)

def get_latest_csv(symbol):
    """Find the latest CSV for a symbol."""
    files = sorted(data_directory.glob(f"{symbol}_*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV found for {symbol}")
    return files[-1]

def preprocess_symbol(symbol):
    csv_path = get_latest_csv(symbol)
    df = pd.read_csv(csv_path, index_col=0)

    # Fix date
    df.index = pd.to_datetime(df.index, errors="coerce")

    # Fix numeric types
    cols = ["1. open", "2. high", "3. low", "4. close", "5. volume"]
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Clean data
    df.dropna(inplace=True)

    df.sort_index(inplace=True)

    # Here we applying a feature engineering 
    for days in MA_DAYS:
        df[f"MA_{days}"] = df["4. close"].rolling(days).mean()

    for days in STD_DAYS:
        df[f"STD_{days}"] = df["4. close"].rolling(days).std()

    df["%_change"] = df["4. close"].pct_change()
    df["Lag_1"] = df["4. close"].shift(1)
    df["Lag_2"] = df["4. close"].shift(2)
    df["Lag_3"] = df["4. close"].shift(3)

    df.dropna(inplace=True)  # dropping rows after with NaN with rolling

    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = PROCESSED_DIR / f"{symbol}_processed_{date_str}.csv"
    df.to_csv(filename)
    print(f"Processed data saved → {filename}")
    return df

if __name__ == "__main__":
    combined_df = pd.DataFrame()
    for symbol in SYMBOLS:
        try:
            df_symbol = preprocess_symbol(symbol)
            df_symbol["Symbol"] = symbol
            combined_df = pd.concat([combined_df, df_symbol])
        except Exception as e:
            print(f"[ERROR] Failed to preprocess {symbol}: {e}")

    ## For Combined data 
    # combined_file = PROCESSED_DIR / f"combined_processed_{datetime.now().strftime('%Y-%m-%d')}.csv"
    # combined_df.to_csv(combined_file)
    # print(f"Combined processed dataset saved → {combined_file}")