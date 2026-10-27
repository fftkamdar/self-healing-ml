
from pathlib import Path

import joblib
import pandas as pd

DATA_PATH = Path("data/raw/Telco-Customer-Churn.csv")
MODEL_PATH = Path("models/baseline_churn_pipeline.joblib")

# Load the original data
df = pd.read_csv(DATA_PATH)

# Prepare a small example in the same feature format
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

X = df.drop(columns=["customerID", "Churn"])

# Load the fitted preprocessing + model pipeline
model = joblib.load(MODEL_PATH)

# Predict churn for three sample records
sample = X.head(3)

predictions = model.predict(sample)
probabilities = model.predict_proba(sample)[:, 1]

print("\nSample predictions")
for i, (prediction, probability) in enumerate(
    zip(predictions, probabilities), start=1
):
    label = "Churn" if prediction == 1 else "No Churn"

    print(
        f"Customer {i}: {label}, "
        f"estimated churn probability = {probability:.3f}"
    )