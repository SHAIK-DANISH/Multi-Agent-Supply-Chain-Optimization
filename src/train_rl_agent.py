import os
from stable_baselines3 import PPO
from inventory_env import InventoryOptimizationEnv

os.makedirs("models", exist_ok=True)

env = InventoryOptimizationEnv()
model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=25000)
model.save("models/inventory_rl_agent")

print("RL Inventory Agent saved to models/inventory_rl_agent.zip")
