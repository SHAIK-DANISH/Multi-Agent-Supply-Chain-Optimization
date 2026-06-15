import os
from pathlib import Path
import pandas as pd
import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

Path("models").mkdir(exist_ok=True)

df = pd.read_csv("data/processed/supply_chain_prepared.csv")

df["product_code"] = df["product_id"].astype("category").cat.codes
df["warehouse_code"] = df["warehouse"].astype("category").cat.codes

X = df[
    [
        "day",
        "product_code",
        "warehouse_code",
        "current_stock",
        "lead_time_days",
        "price",
        "holding_cost",
        "stockout_penalty",
    ]
]

y = df["demand"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = XGBRegressor(
    n_estimators=250,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("Demand Forecasting Model Trained")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

joblib.dump(model, "models/demand_model.pkl")
joblib.dump(
    {
        "product_categories": df["product_id"].astype("category").cat.categories.tolist(),
        "warehouse_categories": df["warehouse"].astype("category").cat.categories.tolist()
    },
    "models/category_mapping.pkl"
)

print("Saved model to models/demand_model.pkl")
