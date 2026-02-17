import pandas as pd
import joblib
import yaml
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Load params
with open("params.yaml") as f:
    params = yaml.safe_load(f)

df = pd.read_csv("data/superstore.csv")

# Feature engineering
df['UnitPrice'] = df['Sales'] / df['Quantity']

features = ['UnitPrice', 'Discount', 'Quantity']
target = 'Profit'

X = df[features]
y = df[target]

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', MLPRegressor(
        hidden_layer_sizes=tuple(params["model"]["hidden_layer_sizes"]),
        max_iter=params["model"]["max_iter"],
        random_state=params["model"]["random_state"]
    ))
])

pipeline.fit(X, y)

joblib.dump(pipeline, "model/model.joblib")

print("✅ Model trained and saved.")
