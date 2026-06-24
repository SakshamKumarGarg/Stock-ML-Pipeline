import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "processed_data"
MODELS_DIR = ROOT_DIR / "models"
CACHE_DIR = ROOT_DIR / "web" / "cache"


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.strftime("%Y-%m-%d")
    return value


def list_symbols():
    symbols = sorted({path.name.split("_")[0] for path in PROCESSED_DIR.glob("*_processed_*.csv")})
    return {"symbols": symbols}


def load_symbol_frame(symbol):
    csv_files = sorted(PROCESSED_DIR.glob(f"{symbol}_processed_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No processed data found for {symbol}")

    df = pd.read_csv(csv_files[-1], index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()
    return df, csv_files[-1]


def build_dashboard(symbol):
    df, csv_path = load_symbol_frame(symbol)
    if "4. close" not in df.columns:
        raise ValueError(f"{symbol} data is missing the '4. close' column")

    df["Target"] = df["4. close"].pct_change().shift(-1)
    df.dropna(inplace=True)

    X = df.drop(columns=["Target"])
    y = df["Target"]

    model_files = sorted(MODELS_DIR.glob(f"{symbol}_*_model.pkl"))
    if not model_files:
        raise FileNotFoundError(f"No trained models found for {symbol}")

    metrics = []
    predictions = {}
    failures = []

    for file_path in model_files:
        model_name = file_path.name.replace(f"{symbol}_", "").replace("_model.pkl", "")

        try:
            model = joblib.load(file_path)
            if hasattr(model, "feature_names_in_"):
                X_input = X[list(model.feature_names_in_)]
            else:
                X_input = X

            y_pred = model.predict(X_input)
            mse = np.mean((y - y_pred) ** 2)
            denominator = np.sum((y - y.mean()) ** 2)
            r2 = 1 - (np.sum((y - y_pred) ** 2) / denominator) if denominator else np.nan
            direction_accuracy = (np.sign(y_pred) == np.sign(y)).mean() * 100

            metrics.append(
                {
                    "model": model_name,
                    "mse": json_safe(mse),
                    "r2": json_safe(r2),
                    "accuracy": json_safe(direction_accuracy),
                }
            )
            predictions[model_name] = y_pred
        except Exception as exc:
            failures.append({"model": model_name, "error": str(exc)})

    if not metrics:
        raise RuntimeError(f"No models could be evaluated for {symbol}")

    metrics.sort(key=lambda item: item["accuracy"] if item["accuracy"] is not None else -1, reverse=True)
    best_model = metrics[0]["model"]

    initial_price = df["4. close"].iloc[0]
    actual_price = (1 + y).cumprod() * initial_price
    predicted_price = pd.Series((1 + predictions[best_model]).cumprod() * initial_price, index=df.index)

    price_points = []
    max_points = 180
    step = max(1, len(df) // max_points)
    sampled_index = list(range(0, len(df), step))
    if sampled_index[-1] != len(df) - 1:
        sampled_index.append(len(df) - 1)

    for idx in sampled_index:
        price_points.append(
            {
                "date": json_safe(df.index[idx]),
                "actual": json_safe(actual_price.iloc[idx]),
                "predicted": json_safe(predicted_price.iloc[idx]),
            }
        )

    recent_rows = []
    for idx, row in df.tail(8).iterrows():
        recent_rows.append(
            {
                "date": json_safe(idx),
                "open": json_safe(row.get("1. open")),
                "high": json_safe(row.get("2. high")),
                "low": json_safe(row.get("3. low")),
                "close": json_safe(row.get("4. close")),
                "volume": json_safe(row.get("5. volume")),
            }
        )

    latest_close = json_safe(df["4. close"].iloc[-1])
    first_close = json_safe(df["4. close"].iloc[0])
    total_return = ((latest_close - first_close) / first_close) * 100 if first_close else None

    return {
        "symbol": symbol,
        "sourceFile": csv_path.name,
        "rows": len(df),
        "dateRange": {
            "start": json_safe(df.index.min()),
            "end": json_safe(df.index.max()),
        },
        "summary": {
            "latestClose": latest_close,
            "totalReturn": json_safe(total_return),
            "bestModel": best_model,
            "bestAccuracy": metrics[0]["accuracy"],
            "modelCount": len(metrics),
        },
        "metrics": metrics,
        "priceSeries": price_points,
        "recentRows": recent_rows,
        "failures": failures,
    }


def export_dashboards():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    symbols = list_symbols()["symbols"]

    for symbol in symbols:
        dashboard = build_dashboard(symbol)
        output_path = CACHE_DIR / f"{symbol}.json"
        output_path.write_text(json.dumps(dashboard, default=json_safe), encoding="utf-8")

    return {"symbols": symbols, "outputDirectory": str(CACHE_DIR)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["symbols", "dashboard", "export"])
    parser.add_argument("--symbol")
    args = parser.parse_args()

    try:
        if args.command == "symbols":
            payload = list_symbols()
        elif args.command == "dashboard":
            if not args.symbol:
                raise ValueError("--symbol is required for dashboard")
            payload = build_dashboard(args.symbol.upper())
        else:
            payload = export_dashboards()

        print(json.dumps({"ok": True, "data": payload}, default=json_safe))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
