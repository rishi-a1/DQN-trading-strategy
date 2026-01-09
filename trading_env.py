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
        price_ser = pd.Series(self.prices)
        vol_ser = pd.Series(self.volumes)

        # Trend
        self.sma_10 = price_ser.rolling(window=10).mean().values
        self.sma_20 = price_ser.rolling(window=20).mean().values
        self.sma_cross = self.sma_10 - self.sma_20

        # Momentum
        delta = price_ser.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        self.rsi_14 = (100 - (100 / (1 + rs))).values
        ema_12 = price_ser.ewm(span=12, adjust=False).mean()
        ema_26 = price_ser.ewm(span=26, adjust=False).mean()
        self.macd = (ema_12 - ema_26).values

        # Volatility
        self.stdev_20 = price_ser.rolling(window=20).std().values

        # Volume
        vol_mean_20 = vol_ser.rolling(window=20).mean()
        self.volume_ratio = (vol_ser / (vol_mean_20 + 1e-8)).values
        self.volume_trend = vol_ser.diff().rolling(window=10).mean().values

        # Price
        self.returns = np.diff(np.log(self.prices), prepend=np.log(self.prices[0]))
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

        self.cash = self.cash - (self.position - prev_position) * self.prices[self.current_step]
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

