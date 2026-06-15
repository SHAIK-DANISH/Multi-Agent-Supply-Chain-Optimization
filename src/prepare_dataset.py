import os
from pathlib import Path
import pandas as pd
import numpy as np

raw_dir = Path("data/raw")
processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)

csv_files = list(raw_dir.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No CSV found in data/raw. Download from Kaggle and place CSV in data/raw folder."
    )

df = pd.read_csv(csv_files[0])

print("Original columns:")
print(df.columns.tolist())

# Normalize column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

# Auto-detect important columns
def find_col(possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

date_col = find_col(["date", "day", "order_date"])
sku_col = find_col(["sku", "product_id", "product", "item_id"])
sales_col = find_col(["sales", "units_sold", "demand", "quantity_sold"])
inventory_col = find_col(["inventory_level", "inventory", "stock", "current_stock"])
lead_time_col = find_col(["supplier_lead_time", "lead_time", "lead_time_days"])
region_col = find_col(["region", "warehouse", "location"])

if sales_col is None:
    raise ValueError("Could not find sales/demand column in Kaggle dataset.")

if sku_col is None:
    df["sku"] = "SKU_" + df.index.astype(str)
    sku_col = "sku"

if date_col is None:
    df["date"] = pd.date_range(start="2024-01-01", periods=len(df), freq="D")
    date_col = "date"

if inventory_col is None:
    df["inventory_level"] = np.random.randint(100, 800, size=len(df))
    inventory_col = "inventory_level"

if lead_time_col is None:
    df["supplier_lead_time"] = np.random.randint(2, 12, size=len(df))
    lead_time_col = "supplier_lead_time"

if region_col is None:
    df["warehouse"] = "WH_DEFAULT"
    region_col = "warehouse"

# Prepare standard project dataset
prepared = pd.DataFrame()
prepared["date"] = pd.to_datetime(df[date_col])
prepared["day"] = (prepared["date"] - prepared["date"].min()).dt.days + 1
prepared["product_id"] = df[sku_col].astype(str)
prepared["warehouse"] = df[region_col].astype(str)
prepared["demand"] = pd.to_numeric(df[sales_col], errors="coerce").fillna(0)
prepared["current_stock"] = pd.to_numeric(df[inventory_col], errors="coerce").fillna(0)
prepared["lead_time_days"] = pd.to_numeric(df[lead_time_col], errors="coerce").fillna(5)

# Add realistic cost fields if not available
prepared["price"] = np.random.randint(500, 1500, size=len(prepared))
prepared["holding_cost"] = np.random.uniform(2, 8, size=len(prepared)).round(2)
prepared["stockout_penalty"] = np.random.randint(50, 150, size=len(prepared))
prepared["warehouse_capacity"] = prepared["current_stock"] + np.random.randint(500, 1500, size=len(prepared))

prepared = prepared.dropna()

prepared.to_csv(processed_dir / "supply_chain_prepared.csv", index=False)

# Supplier table for Supplier Agent
products = prepared["product_id"].drop_duplicates().head(100).tolist()
suppliers = ["SUP_A", "SUP_B", "SUP_C", "SUP_D"]

supplier_rows = []
for product in products:
    for supplier in suppliers:
        supplier_rows.append({
            "supplier": supplier,
            "product_id": product,
            "unit_cost": np.random.randint(250, 900),
            "lead_time_days": np.random.randint(2, 12),
            "reliability_score": round(np.random.uniform(0.75, 0.98), 2)
        })

pd.DataFrame(supplier_rows).to_csv(processed_dir / "supplier_data.csv", index=False)

inventory = prepared[
    ["product_id", "warehouse", "current_stock", "warehouse_capacity"]
].drop_duplicates(["product_id", "warehouse"])

inventory.to_csv(processed_dir / "inventory_data.csv", index=False)

print("Prepared dataset saved to data/processed/supply_chain_prepared.csv")
print("Supplier data saved to data/processed/supplier_data.csv")
print("Inventory data saved to data/processed/inventory_data.csv")
