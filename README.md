# Multi-Agent Supply Chain Optimization Using Kaggle Dataset

This project uses a Kaggle supply chain inventory dataset instead of fully synthetic data.

Recommended Dataset:
High-Dimensional Supply Chain Inventory Dataset
Kaggle slug:
ziya07/high-dimensional-supply-chain-inventory-dataset

## Run Steps

```bash
pip install -r requirements.txt

python src/download_kaggle_dataset.py
python src/prepare_dataset.py
python src/demand_forecasting.py
python src/train_rl_agent.py
streamlit run app/dashboard.py
```

## Notes

If Kaggle download fails, manually download the CSV from Kaggle and place it in:

```bash
data/raw/supply_chain_dataset1.csv
```

Then run:

```bash
python src/prepare_dataset.py
```
