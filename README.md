# Stock Market Prediction ML Workflow

An end-to-end machine learning workflow for fetching stock data, engineering
time-series features, training multiple regression models, evaluating their
performance, and exploring the results in an interactive web dashboard.

## Features

- Fetches market data with `yfinance`
- Preprocesses data and creates rolling time-series features
- Predicts next-period returns instead of raw prices
- Trains and compares multiple machine learning models
- Reports MSE, R2, MAPE, and direction accuracy
- Stores processed CSV files, trained models, and evaluation reports
- Includes an Express, HTML, CSS, and JavaScript web dashboard
- Keeps the original Streamlit dashboard as an optional interface

## Supported Stocks

The current configuration includes AAPL, AMZN, GOOGL, META, MSFT, and NFLX.
Edit the `symbols` list in `config.yaml` to change the pipeline stocks.

## Prerequisites

- Python 3.10 or newer
- Node.js 18 or newer
- npm

## Python Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux or macOS, activate the environment with:

```bash
source .venv/bin/activate
```

## Run the ML Pipeline

From the project root:

```bash
python src/auto_pipeline.py
```

| Directory | Contents |
| --- | --- |
| `data/` | Raw stock data |
| `processed_data/` | Cleaned data and engineered features |
| `models/` | Trained Joblib model files |
| `reports/` | Model evaluation reports |

## Run the Web Dashboard

Install the frontend dependencies once:

```powershell
cd web
npm install
```

Start the application:

```powershell
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in a browser. Keep the
terminal open while using the dashboard and press `Ctrl+C` to stop it.

The `predev` script runs `python ../src/dashboard_api.py export` before Express
starts. It evaluates the saved models and creates JSON files in `web/cache/`.
Express serves these files when a stock is selected.

Restart `npm run dev` whenever processed data or trained models change so the
dashboard cache is regenerated.

## Optional Streamlit Dashboard

```bash
streamlit run dashboard.py
```

## Web Application Structure

| File | Purpose |
| --- | --- |
| `web/server.js` | Express server and JSON API routes |
| `web/public/index.html` | Dashboard page structure |
| `web/public/styles.css` | Responsive dashboard styling |
| `web/public/app.js` | Stock selection, API calls, tables, and chart rendering |
| `src/dashboard_api.py` | Model evaluation and cache generation |

API routes:

- `GET /api/symbols`
- `GET /api/dashboard/:symbol`

## Project Structure

```text
ml-workflow/
|-- data/
|-- processed_data/
|-- models/
|-- reports/
|-- src/
|   |-- auto_pipeline.py
|   |-- dashboard_api.py
|   |-- data_fetch.py
|   |-- preprocess.py
|   `-- train_model.py
|-- web/
|   |-- public/
|   |   |-- app.js
|   |   |-- index.html
|   |   `-- styles.css
|   |-- package.json
|   `-- server.js
|-- config.yaml
|-- dashboard.py
|-- requirements.txt
`-- README.md
```

## Troubleshooting

If the stock dropdown is empty or shows stale content:

1. Stop the server with `Ctrl+C`.
2. Run `npm run dev` again from the `web` directory.
3. Refresh the browser with `Ctrl+Shift+R`.

You can verify the available stocks at:

```text
http://localhost:3000/api/symbols
```
