# 🧠 StockMLPipeline

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![GitHub Actions](https://img.shields.io/github/actions/workflow/status/yourusername/StockMLPipeline/pipeline.yml?label=GitHub%20Actions)](https://github.com/yourusername/StockMLPipeline/actions)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Repo Size](https://img.shields.io/github/repo-size/yourusername/StockMLPipeline.svg)](https://github.com/yourusername/StockMLPipeline)

> ⚙️ *A modular and automated Machine Learning pipeline for stock data — from fetching and preprocessing to model training and reporting.*

---

## 📘 Overview

**StockMLPipeline** is a configurable, per-symbol ML pipeline that automates:
- Fetching real stock data from **Alpha Vantage**
- Cleaning, preprocessing, and feature generation
- Training ML models for each stock symbol
- Saving performance metrics, models, and plots
- Optionally running the entire process automatically via **GitHub Actions**

The project demonstrates practical ML engineering and lightweight MLOps skills — perfect for showcasing workflow automation and structured ML design.

---

## 🔄 Pipeline Phases

| Phase | Description |
|-------|--------------|
| **Phase 1** | Fetch stock data using the Alpha Vantage API (one CSV per symbol). |
| **Phase 2** | Preprocess data (handle NaNs, compute moving averages, volatility features). |
| **Phase 3** | Train ML models per symbol (e.g., RandomForest) and evaluate with MSE/R². |
| **Phase 4** | Automate the entire process using GitHub Actions with daily report updates. |

---

## 🧩 Tech Stack

| Category | Tools |
|-----------|-------|
| **Language** | Python 3.12 |
| **Libraries** | `pandas`, `scikit-learn`, `matplotlib`, `pyyaml`, `joblib` |
| **Automation** | GitHub Actions |
| **Data Source** | Alpha Vantage API |

---

## ⚙️ Setup (Local Execution)

### 1️⃣ Clone the repo
```
git clone https://github.com/SakshamKumarGarg/Stock-ML-Pipeline.git
```

### 2️⃣ Create virtual environment
```
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Linux/Mac
source .venv/bin/activate
```

### 3️⃣ Install dependencies
```
pip install -r requirements.txt
```

### 4️⃣ Add your API key
Create a .env file in the root folder:
```
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

### 5️⃣ Run the full pipeline
```
python src/auto_pipeline.py
```

📁 Repository Structure
```
StockMLPipeline/
├── data/                # Raw stock data (sample/demo)
├── processed_data/       # Cleaned/preprocessed CSVs
├── models/              # Trained ML models (demo)
├── reports/             # Generated plots & reports
├── src/
│   ├── data_fetch.py
│   ├── preprocess.py
│   ├── train_model.py
│   └── auto_pipeline.py
├── config.yaml           # Symbols, model type, parameters
├── requirements.txt
├── .github/workflows/pipeline.yml
└── README.md
```

📊 Outputs
| Folder            | Description                                  |
| ----------------- | -------------------------------------------- |
| `data/`           | Raw fetched data per symbol                  |
| `processed_data/` | Cleaned data with features                   |
| `models/`         | Trained model files                          |
| `reports/`        | Generated plots and performance text reports |


