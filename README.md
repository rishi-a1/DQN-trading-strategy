# Wobbles — A Vanilla DQN Single-Asset Trading Agent

> **Status: Failed experiment. Kept public because the failure modes are instructive.**

Wobbles is a reinforcement learning agent that learns to trade SPY (S&P 500 ETF) at daily frequency using a vanilla Deep Q-Network. The goal was never to generate deployable alpha — it was to understand what happens when you naively apply a classic RL algorithm to a financial time series.

---

## What It Does

The agent frames trading as a Markov Decision Process. At each daily timestep it observes market state and decides to either go flat (hold no position) or go long (hold one share). It learns this policy through DQN trained on 4 years of SPY data (2020–2024).

### MDP Formulation

**State (36 dimensions)**
- Current position (0 or 1)
- Trend: SMA-20 (normalised), SMA-10/20 crossover signal
- Momentum: RSI-14, MACD (both normalised)
- Volatility: 20-day rolling std (normalised)
- Price: 10-day window of log returns
- Volume: 10-day window of volume ratio (vs 20-day MA) and volume trend

**Actions**
- `0` — Go flat (exit position)
- `1` — Go long (hold one share)

**Reward**
```
reward = pnl_log_return
       - transaction_cost_penalty
       - holding_penalty          # small nudge against indefinite flat
       - drawdown_penalty         # 0.1 × current drawdown from peak
       - volatility_penalty       # discourages holding in high-vol regimes
```

**Discount factor:** γ = 0.99

### Network Architecture

```
Input (36) → Linear(128) → ReLU → Linear(64) → ReLU → Linear(2)
```

Standard DQN setup: policy network + frozen target network with soft updates (τ = 0.005), ε-greedy exploration decaying from 1.0 → 0.01, experience replay buffer (capacity 50,000), Huber loss, AdamW optimiser (lr = 3e-4).

### Training Setup

- **Data:** SPY daily adjusted close + volume, 2020-01-01 to 2024-01-01 (~1000 trading days)
- **Episodes:** 100
- **Hardware:** Apple MPS (M-series GPU)
- **Batch size:** 128

---

## Project Structure

```
├── Wobbles_dqn_agent.py   # DQN agent, training loop, replay buffer
├── Ctrading_env.py        # Custom Gym environment
├── backtest.py            # Evaluation against buy-and-hold benchmark
├── utils/                 # Helper utilities
└── backtest_results.png   # Output chart (P&L, drawdown, action distribution)
```

---

## Usage

```bash
pip install torch yfinance gymnasium numpy pandas matplotlib

# Train
python Wobbles_dqn_agent.py

# Backtest on held-out period (2024-01-01 to 2024-07-01)
python backtest.py
```

The trained model is saved to `/Networks/dqn_trading_policy_net.pth`. Update `MODEL_PATH` in `backtest.py` if you save it elsewhere.

---

## Why It Failed

### 1. The Black Box Problem in Finance

DQN is fundamentally uninterpretable - we don't know if there is any true thesis behind its actions and, to add on, the replay buffer confuses it further which I touch on next.

### 2. The Replay Buffer Was the Wrong Tool Here

Comes down to the fact that a buy/hold/sell signal can change depending on the time so the replay buffer just isn't flexible enough for financial markets. The consequence is that whatever the agent learns is likely specific 2020–2024 SPY trajectory rather than a genuine edge.

### 3. 100 Episodes Is Not Enough

DQN on Atari trains for tens of millions of environment steps. This agent trains for roughly 100,000 steps (100 episodes × ~1000 days). It's also training on a single asset with a single historical path. The agent simply doesn't see enough variation to learn anything robust.

### 4. SPY Is the Wrong Target

SPY is too efficient - a 128→64→2 network trained on 4 years of daily data has no plausible edge here. Perhaps a less efficient market would have allowed for an edge.

---

## What Actually Worked 

The engineering is clean thanks to documentation given by Gym:

- The custom Gym environment is properly structured and reusable — you can drop in any single-asset price/volume series and it just works
- The reward shaping (transaction cost + drawdown + vol penalty) feels logical, though the approach may have been flawed
- The backtest separates training and evaluation periods correctly and computes real metrics (Sharpe, max drawdown, action distribution)

---

## Dependencies

```
torch
gymnasium
yfinance
numpy
pandas
matplotlib
```

---

## Acknowledgments

Architecture follows the [PyTorch DQN tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html). Environment structure borrows from OpenAI Gym conventions.
