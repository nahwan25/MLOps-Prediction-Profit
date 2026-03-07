import json
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score

# ===============================
# 1️⃣ Load Model
# ===============================
model = joblib.load("model/model.joblib")

# ===============================
# 2️⃣ Load Data
# ===============================
df = pd.read_csv("data/superstore.csv")

# ===============================
# 3️⃣ Feature Engineering
# ===============================
df["UnitPrice"] = df["Sales"] / df["Quantity"]

# Handle infinite values
df.replace([float("inf"), -float("inf")], pd.NA, inplace=True)

# Drop rows where target missing
df = df.dropna(subset=["Profit"])

features = ["UnitPrice", "Discount", "Quantity"]
target = "Profit"

X = df[features]
y = df[target]

# ===============================
# 4️⃣ Predict
# ===============================
y_pred = model.predict(X)

# ===============================
# 5️⃣ Metrics
# ===============================
metrics = {
    "MAE": mean_absolute_error(y, y_pred),
    "R2": r2_score(y, y_pred)
}

print(metrics)

# ===============================
# 6️⃣ Save metrics
# ===============================
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print("✅ Evaluation completed!")
