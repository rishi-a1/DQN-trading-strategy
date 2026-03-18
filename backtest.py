"""
DQN Trading Strategy Backtest
==============================
Loads the trained policy_net weights and runs the agent on a
held-out test period, then prints / plots performance metrics.

Usage:
    python backtest_dqn.py

Requirements: torch, yfinance, numpy, pandas, matplotlib
Make sure trading_env.py is in the same directory as this file.
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.nn.functional as F
import yfinance as yf

from Ctrading_env import trading_env

# ── Hyperparameters (must match training) ──────────────────────────────────
MODEL_PATH   = "dqn_trading_policy_net.pth"  # adjust path if needed
TICKER       = "SPY"
TRAIN_END    = "2020-01-01"   # same end used in training
TEST_START   = "2024-01-01"   # held-out window (4 years inside the trained range)
TEST_END     = "2024-07-01"
START_STEP   = 0              # portfolio_value starts at 0 (1-share P&L mode)

device = torch.device("cpu")  # inference on CPU is fine

# ── Model architecture (must match DQN in Wobbles_dqn_agent.py) ────────────
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


# ── Data download ───────────────────────────────────────────────────────────
print(f"Downloading {TICKER} data ({TEST_START} → {TEST_END}) ...")
raw = yf.download(
    TICKER,
    start=TEST_START,
    end=TEST_END,
    group_by="column",
    auto_adjust=False,
    multi_level_index=False,
)
price_data = raw["Adj Close"].values.astype(np.float32)
volume_data = raw["Volume"].values.astype(np.float32)
dates = raw.index

print(f"  {len(price_data)} trading days loaded.\n")

# ── Environment & model ─────────────────────────────────────────────────────
env = trading_env(price_data, volume_data)
n_actions = env.action_space.n
state, _ = env.reset()
n_obs = len(state)

policy_net = DQN(n_obs, n_actions).to(device)
policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
policy_net.eval()
print(f"Model loaded from  '{MODEL_PATH}'  ({n_obs} inputs → {n_actions} actions)\n")

# ── Greedy rollout ──────────────────────────────────────────────────────────
state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

portfolio_values = [0.0]
actions_taken    = []
rewards_list     = []

with torch.no_grad():
    while True:
        action = policy_net(state_tensor).max(1).indices.item()
        obs, reward, terminated, truncated, _ = env.step(action)
        actions_taken.append(action)
        rewards_list.append(reward)
        portfolio_values.append(env.portfolio_value)

        if terminated or truncated:
            break
        state_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

# ── Buy-and-hold benchmark ──────────────────────────────────────────────────
start_price = price_data[env.window_size]
bah_values  = [p - start_price for p in price_data[env.window_size:]]

# Align lengths
min_len = min(len(portfolio_values), len(bah_values))
portfolio_values = portfolio_values[:min_len]
bah_values = bah_values[:min_len]
step_dates = dates[env.window_size : env.window_size + min_len]

# ── Metrics ─────────────────────────────────────────────────────────────────
def compute_metrics(values, label=""):
    vals = np.array(values)
    total = vals[-1]
    rets = np.diff(vals)
    vol = rets.std() * math.sqrt(252)
    sharpe = (rets.mean() / (rets.std() + 1e-9)) * math.sqrt(252)
    peak = np.maximum.accumulate(vals)
    dd = vals - peak
    mdd = dd.min()
    print(f"{'─'*40}")
    print(f"  {label}")
    print(f"  Total P&L        : ${total:+.2f}")
    print(f"  Daily vol        : ${vol:.4f}")
    print(f"  Sharpe ratio     : {sharpe:.3f}")
    print(f"  Max Drawdown     : ${mdd:.2f}")
    return dd

print("\n========== BACKTEST RESULTS ==========")
dqn_dd  = compute_metrics(portfolio_values, "DQN Agent")
bah_dd  = compute_metrics(bah_values,       "Buy & Hold")
print("─"*40)

action_labels = {0: "Short", 1: "Hold", 2: "Long"}
counts = {action_labels[k]: actions_taken.count(k) for k in range(3)}
print("\nAction distribution:")
for k, v in counts.items():
    pct = 100 * v / len(actions_taken)
    print(f"  {k:6s}: {v:5d}  ({pct:.1f}%)")
print()

# ── Plot ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45)

# 1. P&L
ax1 = fig.add_subplot(gs[0])
ax1.plot(step_dates, portfolio_values, label="DQN Agent",   color="#2196F3", linewidth=1.5)
ax1.plot(step_dates, bah_values,       label="Buy & Hold",  color="#FF9800", linewidth=1.5, linestyle="--")
ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax1.set_title(f"{TICKER} — P&L per Share  ({TEST_START} to {TEST_END})", fontsize=13)
ax1.set_ylabel("P&L ($)")
ax1.legend()
ax1.grid(alpha=0.3)

# 2. Drawdown
ax2 = fig.add_subplot(gs[1])
ax2.fill_between(step_dates, dqn_dd, 0, alpha=0.4, color="#2196F3", label="DQN")
ax2.fill_between(step_dates, bah_dd, 0, alpha=0.3, color="#FF9800", label="Buy & Hold")
ax2.set_title("Drawdown ($ per share)", fontsize=12)
ax2.set_ylabel("Drawdown ($)")
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Actions taken
action_colors = {0: "#F44336", 1: "#9E9E9E", 2: "#4CAF50"}
action_arr    = np.array(actions_taken)
n_act         = len(action_arr)
act_dates     = step_dates[1:n_act+1] if len(step_dates) > n_act else step_dates[:n_act]

ax3 = fig.add_subplot(gs[2])
for a, color in action_colors.items():
    mask = action_arr == a
    ax3.scatter(
        act_dates[mask] if len(act_dates) == len(action_arr) else np.arange(n_act)[mask],
        np.full(mask.sum(), a),
        c=color, s=4, label=action_labels[a]
    )
ax3.set_yticks([0, 1, 2])
ax3.set_yticklabels(["Short", "Hold", "Long"])
ax3.set_title("Actions Over Time", fontsize=12)
ax3.legend(markerscale=3)
ax3.grid(alpha=0.2)

plt.savefig("backtest_results.png", dpi=150, bbox_inches="tight")
print("Chart saved to  backtest_results.png")
plt.show()