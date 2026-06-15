import pandas as pd

class WarehouseAgent:
    def __init__(self, inventory_file="data/processed/inventory_data.csv"):
        self.inventory = pd.read_csv(inventory_file)

    def check_capacity(self, product_id, warehouse, reorder_quantity):
        row = self.inventory[
            (self.inventory["product_id"].astype(str) == str(product_id)) &
            (self.inventory["warehouse"].astype(str) == str(warehouse))
        ]

        if row.empty:
            return {
                "status": "warehouse/product not found",
                "can_store": False
            }

        row = row.iloc[0]
        future_stock = row["current_stock"] + reorder_quantity

        return {
            "product_id": product_id,
            "warehouse": warehouse,
            "current_stock": int(row["current_stock"]),
            "warehouse_capacity": int(row["warehouse_capacity"]),
            "future_stock": int(future_stock),
            "can_store": bool(future_stock <= row["warehouse_capacity"])
        }
