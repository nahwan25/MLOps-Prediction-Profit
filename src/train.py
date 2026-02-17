import os
import pandas as pd
import joblib
import yaml
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ===============================
# 1️⃣ Load Parameters
# ===============================
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# ===============================
# 2️⃣ Load Dataset
# ===============================
df = pd.read_csv("data/superstore.csv")

# ===============================
# 3️⃣ Feature Engineering
# ===============================
df["UnitPrice"] = df["Sales"] / df["Quantity"]

# Replace infinite values with NaN
df.replace([float("inf"), -float("inf")], pd.NA, inplace=True)

# Drop rows where target is missing
df = df.dropna(subset=["Profit"])

# ===============================
# 4️⃣ Define Features & Target
# ===============================
features = ["UnitPrice", "Discount", "Quantity"]
target = "Profit"

X = df[features]
y = df[target]

# Debug (optional, bisa hapus nanti)
print("Missing values in X:\n", X.isna().sum())
print("Missing values in y:\n", y.isna().sum())

# ===============================
# 5️⃣ Build Pipeline
# ===============================
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("model", MLPRegressor(
        hidden_layer_sizes=tuple(params["model"]["hidden_layer_sizes"]),
        max_iter=params["model"]["max_iter"],
        random_state=params["model"]["random_state"]
    ))
])

# ===============================
# 6️⃣ Train Model
# ===============================
pipeline.fit(X, y)

# ===============================
# 7️⃣ Save Model
# ===============================
os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/model.joblib")

print("✅ Model trained and saved successfully!")
