import os
import shutil
from pathlib import Path
import kagglehub

DATASET_SLUG = "ziya07/high-dimensional-supply-chain-inventory-dataset"

raw_dir = Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)

print("Downloading Kaggle dataset...")
dataset_path = kagglehub.dataset_download(DATASET_SLUG)
dataset_path = Path(dataset_path)

print("Dataset downloaded to:", dataset_path)

csv_files = list(dataset_path.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV file found in downloaded Kaggle dataset.")

for file in csv_files:
    destination = raw_dir / file.name
    shutil.copy(file, destination)
    print("Copied:", destination)

print("Kaggle dataset ready in data/raw/")
