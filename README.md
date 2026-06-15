# 🚀 Multi-Agent Supply Chain Optimization Using AI & Reinforcement Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost-green.svg)]()
[![Reinforcement Learning](https://img.shields.io/badge/Reinforcement%20Learning-PPO-orange.svg)]()
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)]()

## 📌 Project Overview

The Multi-Agent Supply Chain Optimization System is an AI-powered decision support platform designed to improve inventory management, supplier selection, warehouse utilization, and demand forecasting.

The system combines Machine Learning, Reinforcement Learning, and Multi-Agent concepts to automate and optimize critical supply chain operations.

This project uses real-world supply chain data from Kaggle and demonstrates how intelligent agents can collaborate to reduce costs, prevent stockouts, and improve operational efficiency.

---

## 🎯 Problem Statement

Modern supply chains face several challenges:

- Demand uncertainty
- Inventory shortages
- Overstocking
- Supplier delays
- High holding costs
- Poor warehouse utilization

Traditional rule-based systems often fail to adapt to changing business conditions.

This project solves these problems using AI-driven autonomous agents that learn optimal decisions from historical supply chain data.

---

## 🎯 Objectives

- Forecast future product demand
- Optimize inventory replenishment
- Recommend reliable suppliers
- Monitor warehouse capacity
- Reduce stockouts
- Reduce inventory holding costs
- Improve supply chain responsiveness

---

# 🏗 System Architecture

```text
                 ┌───────────────────┐
                 │ Kaggle Dataset    │
                 └─────────┬─────────┘
                           │
                           ▼
                 ┌───────────────────┐
                 │ Data Preparation  │
                 └─────────┬─────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼

 ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
 │ Demand      │   │ Supplier    │   │ Warehouse   │
 │ Forecasting │   │ Agent       │   │ Agent       │
 │ Agent       │   │             │   │             │
 └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼

             ┌────────────────────────┐
             │ Inventory RL Agent     │
             │ (PPO Reinforcement RL) │
             └──────────┬─────────────┘
                        ▼

             ┌────────────────────────┐
             │ Streamlit Dashboard    │
             └────────────────────────┘
```

---

# 🤖 Multi-Agent Components

## 1️⃣ Demand Forecasting Agent

Predicts future demand using historical sales data.

### Algorithms

- XGBoost
- Random Forest
- Gradient Boosting

### Inputs

- Product ID
- Historical Sales
- Inventory Level
- Lead Time

### Outputs

- Predicted Demand
- Demand Trend Analysis
- Reorder Suggestions

---

## 2️⃣ Inventory Optimization Agent

Uses Reinforcement Learning to determine optimal reorder quantities.

### Algorithm

- PPO (Proximal Policy Optimization)

### State Space

- Current Stock
- Demand
- Holding Cost
- Lead Time
- Stockout Penalty

### Actions

- Reorder 0 Units
- Reorder 50 Units
- Reorder 100 Units
- Reorder 150 Units
- Reorder 200 Units

### Reward Function

```text
Reward =
Revenue
- Purchase Cost
- Holding Cost
- Stockout Cost
- Lead Time Cost
```

---

## 3️⃣ Supplier Selection Agent

Evaluates suppliers using:

- Unit Cost
- Reliability Score
- Lead Time

### Output

Recommended Supplier with highest performance score.

---

## 4️⃣ Warehouse Management Agent

Monitors:

- Current Stock
- Warehouse Capacity
- Storage Utilization

### Output

- Capacity Alerts
- Storage Recommendations
- Reorder Feasibility

---

# 📊 Dataset

### Source

Kaggle

**High-Dimensional Supply Chain Inventory Dataset**

Dataset Link:

https://www.kaggle.com/datasets/ziya07/high-dimensional-supply-chain-inventory-dataset

### Features Used

| Feature | Description |
|----------|------------|
| Product ID | Unique Product Identifier |
| Demand | Sales Quantity |
| Inventory Level | Current Stock |
| Lead Time | Supplier Delivery Time |
| Region | Warehouse Location |
| Replenishment | Inventory Refill Data |

---

# 🛠 Technologies Used

## Programming Language

- Python

## Machine Learning

- Scikit-Learn
- XGBoost

## Reinforcement Learning

- Stable-Baselines3
- Gymnasium

## Dashboard

- Streamlit
- Plotly

## Data Processing

- Pandas
- NumPy

## Model Storage

- Joblib

---

# 📂 Project Structure

```text
multi-agent-supply-chain/

│
├── app/
│   └── dashboard.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── demand_model.pkl
│   └── inventory_rl_agent.zip
│
├── src/
│   ├── download_kaggle_dataset.py
│   ├── prepare_dataset.py
│   ├── demand_forecasting.py
│   ├── inventory_env.py
│   ├── supplier_agent.py
│   ├── warehouse_agent.py
│   └── train_rl_agent.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# ⚙ Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/multi-agent-supply-chain.git

cd multi-agent-supply-chain
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📥 Download Kaggle Dataset

```bash
python src/download_kaggle_dataset.py
```

OR

Manually download the dataset and place it inside:

```text
data/raw/
```

---

# 📈 Data Preparation

```bash
python src/prepare_dataset.py
```

Output:

```text
data/processed/supply_chain_prepared.csv
data/processed/supplier_data.csv
data/processed/inventory_data.csv
```

---

# 🧠 Train Demand Forecasting Model

```bash
python src/demand_forecasting.py
```

Example Output

```text
Demand Forecasting Model Trained

MAE : 5.41

RMSE : 7.23
```

---

# 🎮 Train Reinforcement Learning Agent

```bash
python src/train_rl_agent.py
```

Output

```text
inventory_rl_agent.zip
```

---

# 📊 Launch Dashboard

```bash
streamlit run app/dashboard.py
```

Dashboard Features:

✅ Demand Trends

✅ Inventory Monitoring

✅ Supplier Recommendations

✅ Warehouse Capacity Analysis

✅ Demand Forecasting

✅ Inventory Risk Alerts

---

# 📈 Expected Results

| Metric | Improvement |
|----------|------------|
| Inventory Cost | 15-25% Reduction |
| Stockouts | 20-35% Reduction |
| Warehouse Utilization | 10-20% Improvement |
| Supplier Efficiency | 15% Improvement |

---

# ⭐ If you find this project useful, consider giving it a star.