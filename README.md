# 📈 Stock Market Prediction ML Pipeline

An end-to-end **Machine Learning pipeline** for stock market prediction using multiple models, time-series data, and an interactive dashboard.

---

## 🚀 Features

- 📊 Fetch real-time stock data using **Yahoo Finance (yfinance)**
- 🧹 Data preprocessing & feature engineering
- 🤖 Train multiple ML models:
  - Linear Regression
  - Decision Tree
  - Random Forest
  - Extra Trees
  - Gradient Boosting
  - XGBoost
- 📉 Time-series aware training (no data leakage)
- 📊 Predict **returns instead of price** (realistic ML approach)
- 📈 Evaluation using:
  - MSE
  - R² Score
  - Direction Accuracy (%)
- 🧠 Naive baseline comparison
- 📊 Interactive **Streamlit Dashboard**

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


