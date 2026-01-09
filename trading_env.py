import gym
import numpy as np
from gym import spaces
import pandas as pd

class trading_env(gym.envs):
    def __init__(self, price_data, volumes, window_size=10, transaction_cost=0.001, cash=10000):
        super(trading_env, self).__init__()
        self.portfolio_value = cash
        self.cash = cash
        self.action_space = spaces.Discrete(3)  # Actions in MDP: buy, hold, sell
        self.prices = price_data  # puts the prices of the stock in the environment
        self.volumes = volumes
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.current_step = 0  # defines the time/trading days elapsed since the start of the learning period
        self.observation_space = self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)  # State variables: Price, Position, Log Return window of 5
# -------------------------------------------------------
        # Indicators to be in observation space

        # Trend
        self.sma_10 = self.rolling_mean(self.prices, 10)
        self.sma_20 = self.rolling_mean(self.prices, 20)
        self.sma_cross = self.sma_10 - self.sma_20

        # Momentum
        self.rsi_14 = self.compute_rsi(14)
        self.macd = self.compute_macd(self.prices)

        # Volatility
        self.stdev_20 = self.rolling_std(self.prices, 20)

        # Volume
        vol_mean_20 = self.rolling_mean(self.volumes, 20)
        self.volume_ratio = self.volumes / (vol_mean_20 + 1e-8)
        self.volume_trend = self.rolling_mean(np.diff(self.volumes, prepend=np.nan), 10)

        # Price
        self.returns = np.diff(np.log(price_data), prepend=np.nan)
# -------------------------------------------------------
        # State variables: Position, SMA20, SMA Cross, RSI, MACD, Stdev (Window = 1)
        # Returns, Vol Ratio, Vol Trend (Window = 10)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(36,),
            dtype=np.float32
        )
        self.reset()

# ------------------------------------------------- Environment functions
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = max(26, self.window_size)
        self.position = 0
        self.total_reward = 0
        self.cash = 100000
        self.portfolio_value = self.cash
        return self._get_state(), {}

    def step(self, action):
        prev_position = self.position
        if action == 0:
            self.position = -1
        if action == 1:
            self.position = 0
        elif action == 2:
            self.position = 1

        price_change = self.prices[self.current_step] - self.prices[self.current_step - 1]
        reward = prev_position * price_change
        if self.position != prev_position:
            reward -= self.transaction_cost
            self.cash -= self.transaction_cost
        self.total_reward += reward
        self.current_step += 1
        self.portfolio_value = self.cash + self.position * self.prices[self.current_step]
        done = self.current_step >= len(self.prices) - 1
        return self.get_state(), reward, done, False, {}

    def get_state(self):
        # Getting the windows of price data for the 3 metrics: returns, volume ratio, volume trend - when reset the
        # environment starts at step 26 so MACD and the other metrics can all be calculated without issue
        r_window = self.returns[self.current_step - self.window_size:self.current_step]
        vr_window = self.volume_ratio[self.current_step - self.window_size:self.current_step]
        vt_window = self.volume_trend[self.current_step - self.window_size:self.current_step]

        state = np.concatenate([
            np.array([
                self.position,
                self.sma_20[self.current_step],
                self.sma_cross[self.current_step],
                self.rsi_14[self.current_step],
                self.macd[self.current_step],
                self.stdev_20[self.current_step],
            ]),
            r_window,
            vr_window,
            vt_window
        ])

        return np.nan_to_num(state).astype(np.float32)

    def rolling_mean(self, arr, window):
        return np.mean(arr[self.current_step-window:self.current_step])

    def rolling_std(self, arr, window):
        return np.std(arr[self.current_step - window:self.current_step])

    def compute_rsi(self, n):
        deltas = np.diff(self.prices, prepend=np.nan)
        gains = np.maximum(deltas, 0)
        losses = -np.minimum(deltas, 0)

        avg_gain = self.rolling_mean(gains, n)
        avg_loss = self.rolling_mean(losses, n)

        rs = avg_gain / (avg_loss + 1e-8)
        return 100 - (100 / (1 + rs))

    def ema(self, x, span):
        return pd.Series(x).ewm(span=span, adjust=False).mean().values

    def compute_macd(self, prices):
        ema_12 = self.ema(prices, 12)
        ema_26 = self.ema(prices, 26)
        return ema_12 - ema_26

