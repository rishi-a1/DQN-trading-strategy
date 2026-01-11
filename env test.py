import numpy as np
import yfinance as yf
import pandas as pd
from trading_env import trading_env

# Load or create your price and volume data
# Example with synthetic data:
np.random.seed(42)

ticker_symbol = "WMT"
start_date = "2022-01-01"
end_date = "2026-01-01"


# Download the historical data
data = yf.download(ticker_symbol, start=start_date, end=end_date, group_by='column', auto_adjust=False, multi_level_index=False)
price_data = data['Adj Close']
volume_data = data['Volume']
print(price_data)

# Create the environment
env = trading_env(price_data=price_data, volumes=volume_data)

# Test with random agent
num_episodes = 5

for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    total_reward = 0
    steps = 0

    print(f"\n--- Episode {episode + 1} ---")

    while not done:
        # Random action: 0 (sell), 1 (hold), or 2 (buy)
        action = env.action_space.sample()

        state, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        # Optional: print every N steps to see progress
        if steps % 100 == 0:
            print(f"Step {steps}: Action={action}, Reward={reward:.4f}, Portfolio Value={env.portfolio_value:.2f}")

    print(f"Episode {episode + 1} finished:")
    print(f"  Total steps: {steps}")
    print(f"  Total reward: {total_reward:.4f}")
    print(f"  Final portfolio value: {env.portfolio_value:.2f}")
    print(f"  Final cash: {env.cash:.2f}")
    print(f"  Final position: {env.position}")