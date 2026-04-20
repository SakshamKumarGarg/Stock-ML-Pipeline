import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Paths
PROCESSED_DIR = Path("processed_data")
MODELS_DIR = Path("models")

st.title("📈 Stock Market ML Dashboard")

# Select stock
symbols = [f.name.split("_")[0] for f in PROCESSED_DIR.glob("*_processed_*.csv")]
symbols = list(set(symbols))

symbol = st.selectbox("Select Stock", symbols)

# Load latest data
csv_files = sorted(PROCESSED_DIR.glob(f"{symbol}_processed_*.csv"))
df = pd.read_csv(csv_files[-1], index_col=0)

df.index = pd.to_datetime(df.index, errors="coerce")

df["Target"] = df["4. close"].pct_change().shift(-1)
df.dropna(inplace=True)

X = df.drop(columns=["Target"])
y = df["Target"]

# Load models (FIXED PATTERN)
model_files = list(MODELS_DIR.glob(f"{symbol}_*_model.pkl"))

results = {}
predictions = {}

for file in model_files:
    model_name = file.name.split("_")[1]
    model = joblib.load(file)

    try:
        # ✅ Ensure feature consistency
        if hasattr(model, "feature_names_in_"):
            X_input = X[model.feature_names_in_]
        else:
            X_input = X

        y_pred = model.predict(X_input)

        mse = np.mean((y - y_pred) ** 2)
        r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - y.mean()) ** 2))
        direction_acc = (np.sign(y_pred) == np.sign(y)).mean() * 100

        results[model_name] = {
            "MSE": mse,
            "R2": r2,
            "Accuracy": direction_acc
        }

        predictions[model_name] = y_pred

    except Exception as e:
        st.warning(f"{model_name} failed: {e}")

# Convert to dataframe
results_df = pd.DataFrame(results).T.reset_index()
results_df.columns = ["Model", "MSE", "R2", "Accuracy"]

# Show table
st.subheader("📊 Model Comparison")
st.dataframe(results_df)

# Plot MSE
st.subheader("📉 Error Comparison")
fig, ax = plt.subplots()
ax.bar(results_df["Model"], results_df["MSE"])
plt.xticks(rotation=30)
st.pyplot(fig)

# Best model
best_model = results_df.loc[results_df["Accuracy"].idxmax()]["Model"]
best_acc = results_df["Accuracy"].max()

st.success(f"🏆 Best Model: {best_model}")
st.metric("Best Accuracy", f"{best_acc:.2f}%")

# Prediction plot
st.subheader("📈 Predictions vs Actual")

selected_model = st.selectbox("Choose Model", list(predictions.keys()))

# Convert returns back to price
initial_price = df["4. close"].iloc[0]

actual_price = (1 + y).cumprod() * initial_price
pred_price = (1 + predictions[selected_model]).cumprod() * initial_price

fig2, ax2 = plt.subplots()
ax2.plot(np.array(actual_price), label="Actual Price")
ax2.plot(np.array(pred_price), label="Predicted Price")
ax2.legend()

st.pyplot(fig2)

st.subheader("📊 Actual Stock Price")

fig3, ax3 = plt.subplots()
ax3.plot(df["4. close"], label="Close Price")
ax3.legend()

st.pyplot(fig3)