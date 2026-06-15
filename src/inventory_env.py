import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class InventoryOptimizationEnv(gym.Env):
    def __init__(self, data_file="data/processed/supply_chain_prepared.csv"):
        super().__init__()

        self.df = pd.read_csv(data_file)
        self.action_space = spaces.Discrete(6)

        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([10000, 10000, 50, 500, 60], dtype=np.float32),
            dtype=np.float32
        )

        self.index = 0
        self.max_steps = min(1000, len(self.df) - 1)
        self.stock = float(self.df.iloc[0]["current_stock"])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.index = 0
        row = self.df.iloc[self.index]
        self.stock = float(row["current_stock"])

        obs = np.array([
            self.stock,
            row["demand"],
            row["holding_cost"],
            row["stockout_penalty"],
            row["lead_time_days"]
        ], dtype=np.float32)

        return obs, {}

    def step(self, action):
        reorder_qty = int(action) * 50

        self.index += 1
        row = self.df.iloc[self.index]

        demand = float(row["demand"])
        holding_cost = float(row["holding_cost"])
        stockout_penalty = float(row["stockout_penalty"])
        lead_time = float(row["lead_time_days"])

        available_stock = self.stock + reorder_qty
        sold_units = min(available_stock, demand)
        stockout_units = max(0, demand - available_stock)
        ending_stock = max(0, available_stock - demand)

        revenue = sold_units * 100
        purchase_cost = reorder_qty * 45
        holding_expense = ending_stock * holding_cost
        stockout_cost = stockout_units * stockout_penalty
        lead_time_cost = reorder_qty * lead_time * 0.5

        reward = revenue - purchase_cost - holding_expense - stockout_cost - lead_time_cost

        self.stock = ending_stock

        terminated = self.index >= self.max_steps
        truncated = False

        obs = np.array([
            self.stock,
            demand,
            holding_cost,
            stockout_penalty,
            lead_time
        ], dtype=np.float32)

        info = {
            "reorder_qty": reorder_qty,
            "demand": demand,
            "ending_stock": ending_stock,
            "stockout_units": stockout_units,
            "reward": reward
        }

        return obs, reward, terminated, truncated, info
