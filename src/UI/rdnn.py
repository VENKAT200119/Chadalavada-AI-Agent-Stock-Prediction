import gym
from gym import spaces
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn

# RDNN.3 imports
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback

class TradingEnv(gym.Env):
    """
    Gym environment for trading based on OHLCV data.
    Observation: window of last T bars (OHLCV).
    Action space: 0=SELL, 1=HOLD, 2=BUY.
    Reward: change in portfolio value minus transaction cost and slippage.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, data, window_size=10, transaction_cost=0.001, slippage=0.001):
        super(TradingEnv, self).__init__()
        # RDNN.1.1: store OHLCV data
        self.data = np.array(data, dtype=np.float32)
        self.window_size = window_size
        # RDNN.1.2: cost and slippage settings
        self.transaction_cost = transaction_cost
        self.slippage = slippage

        # Define action/observation spaces
        self.action_space = spaces.Discrete(3)
        obs_shape = (window_size, self.data.shape[1])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)
        self.reset()

    def reset(self):
        # RDNN.1.1: reset environment state
        self.current_step = self.window_size
        self.position = 0
        self.entry_price = 0.0
        self.net_worth = 1.0
        return self._get_observation()

    def _get_observation(self):
        # RDNN.1.1: return last T bars
        return self.data[self.current_step - self.window_size : self.current_step]

    def step(self, action):
        # RDNN.1.2: apply action, compute reward with costs
        done = False
        price = self.data[self.current_step, 3]
        reward = 0.0
        # SELL
        if action == 0 and self.position != -1:
            reward -= self._cost(price)
            self.position = -1
            self.entry_price = price
        # BUY
        elif action == 2 and self.position != 1:
            reward -= self._cost(price)
            self.position = 1
            self.entry_price = price
        # HOLD does nothing

        # Calculate P&L
        pnl = self.position * (price - self.entry_price)
        reward += pnl
        self.net_worth += pnl - self._cost(price)

        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True

        obs = self._get_observation() if not done else np.zeros_like(self._get_observation())
        return obs, reward, done, {"net_worth": self.net_worth}

    def _cost(self, price):
        # RDNN.1.2: simple transaction cost + slippage
        return price * (self.transaction_cost + self.slippage)

    def render(self, mode="human"):
        # RDNN.1.3: render state
        print(f"Step: {self.current_step}, Position: {self.position}, Net worth: {self.net_worth:.3f}")


def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> np.ndarray:
    """
    Fetch OHLCV data with yfinance (RDNN.1).
    Returns array with columns [Open, High, Low, Close, Volume].
    """
    df = yf.download(ticker, period=period, interval=interval)
    df = df.dropna()[["Open", "High", "Low", "Close", "Volume"]]
    return df.values

# ===== RDNN.2: Build RNN Policy Network =====
# RDNN.2.1: LSTM encoder
# RDNN.2.2: DQN & PPO heads
# RDNN.2.3: dummy forward-pass
class RNNPolicyNetwork(nn.Module):
    """
    LSTM-based encoder with DQN and PPO heads.
    """
    def __init__(self, input_size, hidden_size, num_layers, action_dim):
        super(RNNPolicyNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.q_head = nn.Linear(hidden_size, action_dim)
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        lstm_out, hidden_out = self.lstm(x, hidden)
        last = lstm_out[:, -1, :]
        return (
            self.q_head(last),        # Q-values
            self.policy_head(last),   # policy logits
            self.value_head(last),    # state-value
            hidden_out
        )

# ===== RDNN.3: Configure RL Algorithm =====
# RDNN.3.1: integrate RecurrentPPO
# RDNN.3.2: set/document hyperparameters
# RDNN.3.3: TensorBoard/EvalCallback logging

def train_recurrent_ppo(env, total_timesteps=10000, log_dir='./logs/'):
    """
    Train a RecurrentPPO agent on TradingEnv.

    Hyperparameters (RDNN.3.2):
      - learning_rate: 3e-4
      - batch_size: 64
      - n_steps: 256
      - tensorboard_log: log_dir  (RDNN.3.3)
    """
    hyperparams = {
        'learning_rate': 3e-4,
        'batch_size': 64,
        'n_steps': 256,
        'tensorboard_log': log_dir,
    }
    model = RecurrentPPO(
        policy='MlpLstmPolicy',
        env=env,
        verbose=1,
        **hyperparams
    )
    eval_callback = EvalCallback(
        env,
        best_model_save_path='./best_model/',
        log_path=log_dir,
        eval_freq=100_000,
        n_eval_episodes=1
    )  # RDNN.3.3: logging & evaluation
    model.learn(total_timesteps=total_timesteps, tb_log_name='rdnn3', callback=eval_callback)
    model.save('rdnn3_recurrentppo')
    return model

if __name__ == '__main__':
    ticker = input('Enter stock ticker symbol (e.g., AAPL): ').strip().upper()
    # Fetch and run simple demo
    try:
        data_array = fetch_ohlcv(ticker)
        print(f'Fetched {data_array.shape[0]} rows of OHLCV data for {ticker}.')
    except Exception as e:
        print(f'Error fetching data for {ticker}: {e}')
        exit(1)

    # RDNN.1 demonstration
    window_size = 10
    env = TradingEnv(data_array, window_size=window_size)
    obs = env.reset()
    print(f'Initial observation shape: {obs.shape}')
    next_obs, reward, done, info = env.step(1)
    print(f'After one step => reward: {reward:.4f}, net_worth: {info['net_worth']:.4f}')

    # RDNN.2.3: dummy forward-pass test
    dummy_obs = torch.randn(2, window_size, data_array.shape[1])
    net = RNNPolicyNetwork(input_size=data_array.shape[1], hidden_size=64, num_layers=1, action_dim=3)
    q, pi, v, _ = net(dummy_obs)
    print(f'Shapes Q:{q.shape} Pi:{pi.shape} V:{v.shape}')

    # RDNN.3 training
    train_recurrent_ppo(env)