import pandas as pd
import joblib
import numpy as np
import json
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/superstore.csv")

df['UnitPrice'] = df['Sales'] / df['Quantity']

features = ['UnitPrice', 'Discount', 'Quantity']
target = 'Profit'

X = df[features]
y = df[target]

model = joblib.load("model/model.joblib")

y_pred = model.predict(X)

metrics = {
    "MAE": mean_absolute_error(y, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
    "R2": r2_score(y, y_pred)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Evaluation completed.")
