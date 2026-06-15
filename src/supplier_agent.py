import pandas as pd

class SupplierAgent:
    def __init__(self, supplier_file="data/processed/supplier_data.csv"):
        self.suppliers = pd.read_csv(supplier_file)

    def recommend_supplier(self, product_id):
        df = self.suppliers[self.suppliers["product_id"] == product_id].copy()

        if df.empty:
            return {"message": "No supplier found for this product"}

        df["score"] = (
            df["reliability_score"] * 50
            - df["unit_cost"] * 0.03
            - df["lead_time_days"] * 2
        )

        return df.sort_values("score", ascending=False).iloc[0].to_dict()
