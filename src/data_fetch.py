import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path
import yaml
import time



load_dotenv()
API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
with open("config.yaml","r") as f:
    config = yaml.safe_load(f)

SYMBOLS = config.get("symbols",["AAPL"])
fetch_delay = config.get("fetch_delay_seconds",20)

data_directory = Path("data")
data_directory.mkdir(exist_ok=True)

def fetch_stock_data(symbol="AAPL"):
    """Here we are fetching stock data from Alpha Vantage"""
    url = "https://www.alphavantage.co/query"
    params = {

        "function":"TIME_SERIES_DAILY",
        "symbol": symbol,
        "apikey" : API_KEY,
        "outputsize" : "compact"
    }

    r = requests.get(url, params=params)
    data = r.json()


    if "Time Series (Daily)" not in data:
        raise ValueError(f"Error in fetching data: {data}")
    
    df = pd.DataFrame.from_dict(data["Time Series (Daily)"],orient="index",dtype="float")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

def save_data(df,symbol = "AAPL"):
    """Saving the dats with timestamped filename"""
    date_str = datetime.now().strftime("%Y-%m-%d")
    filename = data_directory/f"{symbol}_{date_str}.csv"
    df.to_csv(filename)
    print(f"Data Saved -> {filename}")

if __name__ == "__main__":
    for symbol in SYMBOLS:
        try:
            print(f"[INFO] Fetching data for {symbol}...")
            df = fetch_stock_data(symbol)
            save_data(df, symbol)
            time.sleep(fetch_delay)  # For Avoid API limit
        except Exception as e:
            print(f"[ERROR] Failed to fetch {symbol}: {e}")

