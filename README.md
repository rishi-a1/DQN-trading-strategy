# DQN-trading-strategy
Wobbles: A vanilla DQN single asset trading agent aimed to optimize trades in x on a daily frequency. Main goal is to explore the use of reinforcement learning in
financial markets rather than generate deployable alpha.

MDP structure:
  State variables:
    State Position (Buy, Hold, Sell)
    Log Return Window of 10 past states
    Volatility
  Actions:
    Buy
    Hold
    Sell
  Transition Function:
    DQN is used so the function used is an approximation of the target function using a neural network
  Reward:
    PnL - transaction cost
  Discount Factor:
    0.99

Network structure:

MSE loss used

Data:

Evaluation:

Limitations:

Usage:

Acknowledgments:
