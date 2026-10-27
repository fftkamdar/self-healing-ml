
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models/selected_churn_pipeline.joblib"
DATA_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"

print("=" * 60)
print("VERIFYING THE INTEGRATED SELECTED MODEL")
print("=" * 60)

# 1. Check that the saved model exists.
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

print("\n[1] Model file exists.")

# 2. Load the saved model.
model = joblib.load(MODEL_PATH)

print("[2] Model loaded successfully.")

# 3. Load a few rows for a prediction smoke test.
df = pd.read_csv(DATA_PATH)

# Remove the target and identifier from the input features.
X_sample = df.drop(columns=["Churn", "customerID"]).head(5).copy()

# Match the preparation used during training.
X_sample["TotalCharges"] = pd.to_numeric(
    X_sample["TotalCharges"], errors="coerce"
)

# 4. Generate predictions and probabilities.
predictions = model.predict(X_sample)
probabilities = model.predict_proba(X_sample)[:, 1]

results = pd.DataFrame(
    {
        "predicted_churn": predictions,
        "churn_probability": probabilities,
    }
)

results["predicted_churn_label"] = results[
    "predicted_churn"
].map({0: "No", 1: "Yes"})

print("\n[3] Prediction smoke test completed.")
print(results.to_string(index=False))

print("\nVerification complete.")
print("Note: this is an inference smoke test, not a new performance evaluation.")