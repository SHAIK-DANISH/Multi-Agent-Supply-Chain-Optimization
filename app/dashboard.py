import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

import pandas as pd
import streamlit as st
import plotly.express as px
import joblib
from supplier_agent import SupplierAgent
from warehouse_agent import WarehouseAgent

st.set_page_config(page_title="Kaggle Supply Chain AI", layout="wide")

st.title("Multi-Agent Supply Chain Optimization System")
st.write("Using Kaggle Supply Chain Inventory Dataset")

sales = pd.read_csv(ROOT / "data/processed/supply_chain_prepared.csv")
inventory = pd.read_csv(ROOT / "data/processed/inventory_data.csv")

supplier_agent = SupplierAgent(ROOT / "data/processed/supplier_data.csv")
warehouse_agent = WarehouseAgent(ROOT / "data/processed/inventory_data.csv")

st.sidebar.header("Filters")

product = st.sidebar.selectbox("Product / SKU", sorted(sales["product_id"].astype(str).unique()))
warehouse = st.sidebar.selectbox("Warehouse / Region", sorted(sales["warehouse"].astype(str).unique()))

filtered = sales[
    (sales["product_id"].astype(str) == str(product)) &
    (sales["warehouse"].astype(str) == str(warehouse))
]

if filtered.empty:
    st.error("No data found for selected product and warehouse.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Demand", int(filtered["demand"].sum()))
col2.metric("Average Demand", round(filtered["demand"].mean(), 2))

stock_row = inventory[
    (inventory["product_id"].astype(str) == str(product)) &
    (inventory["warehouse"].astype(str) == str(warehouse))
]

if stock_row.empty:
    current_stock = 0
    capacity = 0
else:
    stock_row = stock_row.iloc[0]
    current_stock = int(stock_row["current_stock"])
    capacity = int(stock_row["warehouse_capacity"])

col3.metric("Current Stock", current_stock)
col4.metric("Warehouse Capacity", capacity)

st.subheader("Demand Trend")
fig = px.line(filtered, x="day", y="demand", title=f"Demand Trend for {product} at {warehouse}")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Supplier Recommendation Agent")
st.json(supplier_agent.recommend_supplier(product))

st.subheader("Warehouse Capacity Agent")
reorder_qty = st.slider("Reorder Quantity", 0, 1000, 200, 50)
st.json(warehouse_agent.check_capacity(product, warehouse, reorder_qty))

st.subheader("Inventory Risk Alert")
avg_demand = filtered["demand"].mean()

if current_stock < avg_demand * 3:
    st.error("High Risk: Stockout possible within 3 days")
elif current_stock < avg_demand * 7:
    st.warning("Medium Risk: Reorder soon")
else:
    st.success("Stock level is healthy")

st.subheader("Demand Forecasting Agent")

try:
    model = joblib.load(ROOT / "models/demand_model.pkl")
    mapping = joblib.load(ROOT / "models/category_mapping.pkl")

    product_categories = mapping["product_categories"]
    warehouse_categories = mapping["warehouse_categories"]

    product_code = product_categories.index(product) if product in product_categories else 0
    warehouse_code = warehouse_categories.index(warehouse) if warehouse in warehouse_categories else 0

    latest = filtered.iloc[-1]
    future_day = int(sales["day"].max()) + 1

    X_future = pd.DataFrame([{
        "day": future_day,
        "product_code": product_code,
        "warehouse_code": warehouse_code,
        "current_stock": current_stock,
        "lead_time_days": latest["lead_time_days"],
        "price": latest["price"],
        "holding_cost": latest["holding_cost"],
        "stockout_penalty": latest["stockout_penalty"]
    }])

    predicted_demand = model.predict(X_future)[0]
    st.metric("Predicted Next-Day Demand", round(float(predicted_demand), 2))

except Exception as e:
    st.warning("Train the demand model first using: python src/demand_forecasting.py")
    st.write(str(e))
