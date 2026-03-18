import gym
import numpy as np
from gym import spaces
import pandas as pd

class trading_env(gym.Env):

    def __init__(self, price_data, volumes, window_size=10, transaction_cost=0.001, cash=0):
        super(trading_env, self).__init__()
        self.portfolio_value = cash
        self.cash = cash
        self.action_space = spaces.Discrete(2)  # Actions in MDP: flat, long
        self.prices = price_data  # puts the prices of the stock in the environment
        self.volumes = volumes  # puts the volumes of the stocks traded in the environment
        self.transaction_cost = transaction_cost  # initialising to value of transaction whatever exchange
        self.window_size = window_size
        self.current_step = 0  # defines the time/trading days elapsed since the start of the learning period
        self.observation_space = self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32)  # State variables: Price, Position, Log Return window of 5
# -------------------------------------------------------
        # Indicators to be in observation space setup
        price_ser = pd.Series(self.prices)
        vol_ser = pd.Series(self.volumes)
        price_mean = np.mean(self.prices)
        price_std = np.std(self.prices)
        # Trend (Normalized)
        self.sma_10 = price_ser.rolling(window=10).mean().values
        self.sma_20 = price_ser.rolling(window=20).mean().values
        self.sma_20 = (self.sma_20 - price_mean) / (price_std + 1e-8)
        self.sma_cross = (self.sma_10 - self.sma_20) / (price_std + 1e-8)

        # Momentum (Normalized)
        delta = price_ser.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-8)
        self.rsi_14 = (100 - (100 / (1 + rs))).values
        self.rsi_14 = (self.rsi_14 - 50) / 50
        ema_12 = price_ser.ewm(span=12, adjust=False).mean()
        ema_26 = price_ser.ewm(span=26, adjust=False).mean()
        self.macd = (ema_12 - ema_26).values
        self.macd = (self.macd - np.nanmean(self.macd)) / (np.nanstd(self.macd) + 1e-8)

        # Volatility (Normalized)
        self.stdev_20 = (price_ser.rolling(window=20).std() / (price_std + 1e-8)).values

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
        self.cash = 0
        self.portfolio_value = self.cash
        self.cumulative_return = 0.0
        self.peak_return = 0.0
        return self.get_state(), {}

    def step(self, action):
        prev_position = self.position
        if action == 0:
            self.position = 0
        elif action == 1:
            self.position = 1

        # --- Portfolio accounting (single transaction cost deduction) ---
        trade_size = abs(self.position - prev_position)
        cost = trade_size * self.transaction_cost * self.prices[self.current_step]
        self.cash = self.cash - (self.position - prev_position) * self.prices[self.current_step] - cost

        # --- Reward components ---
        log_return = np.log(self.prices[self.current_step] / self.prices[self.current_step - 1])

        # 1. Risk-adjusted PnL: reward log return earned by the position held
        pnl_reward = prev_position * log_return

        # 2. Transaction cost penalty (proportional, consistent units with pnl_reward)
        tc_penalty = trade_size * self.transaction_cost

        # 3. Holding penalty: small nudge to discourage indefinite flat positions
        holding_penalty = 0.0001 if self.position == 0 else 0.0

        # 4. Drawdown penalty: track cumulative log-return drawdown (price-normalised, no cash blow-up)
        self.portfolio_value = self.cash + self.position * self.prices[self.current_step]
        self.cumulative_return += pnl_reward
        self.peak_return = max(self.peak_return, self.cumulative_return)
        drawdown = self.peak_return - self.cumulative_return  # always >= 0, same units as pnl_reward
        drawdown_penalty = 0.1 * drawdown

        # 5. Volatility penalty: discourage holding large positions in high-volatility regimes
        recent_vol = self.stdev_20[self.current_step]
        vol_penalty = 0.05 * abs(self.position) * max(recent_vol - 1.0, 0.0)

        reward = pnl_reward - tc_penalty - holding_penalty - drawdown_penalty - vol_penalty

        self.total_reward += reward
        self.current_step += 1
        done = self.current_step >= len(self.prices) - 1
        return self.get_state(), reward, done, False, {}

    def get_state(self):
        # Getting the windows of price data for the 3 metrics: returns, volume ratio, volume trend - when reset the
        # environment starts at step 26 so MACD and the other metrics can all be calculated without issue
        start = self.current_step - self.window_size + 1
        end = self.current_step+1
        r_window = self.returns[start:end]
        vr_window = self.volume_ratio[start:end]
        vt_window = self.volume_trend[start:end]

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