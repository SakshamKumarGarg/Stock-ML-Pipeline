import pandas as pd
from datetime import datetime
from pathlib import Path
import yfinance as yf
import yaml

# Load config
with open("config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

SYMBOLS = config.get("symbols", ["AAPL"])

data_directory = Path("data")
data_directory.mkdir(exist_ok=True)


def fetch_stock_data(symbol="AAPL"):
    """Fetch stock data using yfinance"""
    df = yf.download(symbol, period="1y", interval="1d")

    if df.empty:
        raise ValueError(f"No data for {symbol}")

    # Rename columns to match your pipeline
    df.rename(columns={
        "Open": "1. open",
        "High": "2. high",
        "Low": "3. low",
        "Close": "4. close",
        "Volume": "5. volume"
    }, inplace=True)

    return df


def save_data(df, symbol="AAPL"):
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = data_directory / f"{symbol}_{date_str}.csv"
    df.to_csv(filename)
    print(f"Data Saved -> {filename}")


if __name__ == "__main__":
    for symbol in SYMBOLS:
        try:
            print(f"[INFO] Fetching data for {symbol}...")
            df = fetch_stock_data(symbol)
            save_data(df, symbol)
        except Exception as e:
            print(f"[ERROR] Failed to fetch {symbol}: {e}")