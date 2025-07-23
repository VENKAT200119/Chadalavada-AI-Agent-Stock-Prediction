import gym
from gym import spaces
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import time
import json

# RDNN.3 imports
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

class TradingEnv(gym.Env):
    """
    Gym environment for trading based on OHLCV data.
    Observation: window of last T bars (OHLCV).
    Action space: 0=SELL, 1=HOLD, 2=BUY.
    Reward: change in portfolio value minus transaction cost and slippage.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, data, window_size=10, transaction_cost=0.001, slippage=0.001):
        super().__init__()
        self.data = np.array(data, dtype=np.float32)
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.slippage = slippage

        self.action_space = spaces.Discrete(3)
        obs_shape = (window_size, self.data.shape[1])
        self.observation_space = spaces.Box(-np.inf, np.inf, obs_shape, dtype=np.float32)
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.position = 0
        self.entry_price = 0.0
        self.net_worth = 1.0
        return self._get_observation()

    def _get_observation(self):
        return self.data[self.current_step - self.window_size : self.current_step]

    def step(self, action):
        done = False
        price = self.data[self.current_step, 3]
        reward = 0.0

        if action == 0 and self.position != -1:
            reward -= self._cost(price)
            self.position = -1
            self.entry_price = price
        elif action == 2 and self.position != 1:
            reward -= self._cost(price)
            self.position = 1
            self.entry_price = price
        # HOLD does nothing

        pnl = self.position * (price - self.entry_price)
        reward += pnl
        self.net_worth += pnl - self._cost(price)

        self.current_step += 1
        if self.current_step >= len(self.data):
            done = True

        obs = self._get_observation() if not done else np.zeros_like(self._get_observation())
        return obs, reward, done, {"net_worth": self.net_worth}

    def _cost(self, price):
        return price * (self.transaction_cost + self.slippage)

    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Position: {self.position}, Net worth: {self.net_worth:.3f}")

def fetch_ohlcv(ticker: str, period: str = "1y", interval: str = "1d") -> np.ndarray:
    df = yf.download(ticker, period=period, interval=interval)
    df = df.dropna()[["Open", "High", "Low", "Close", "Volume"]]
    return df.values

class RNNPolicyNetwork(nn.Module):
    """
    LSTM-based encoder with DQN and PPO heads.
    """
    def __init__(self, input_size, hidden_size, num_layers, action_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.q_head = nn.Linear(hidden_size, action_dim)
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        lstm_out, hidden_out = self.lstm(x, hidden)
        last = lstm_out[:, -1, :]
        return (
            self.q_head(last),
            self.policy_head(last),
            self.value_head(last),
            hidden_out
        )

def train_recurrent_ppo(env, total_timesteps=10000, log_dir='./logs/'):
    model = RecurrentPPO(
        policy='MlpLstmPolicy',
        env=env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=256,
        tensorboard_log=log_dir,
    )
    eval_cb = EvalCallback(
        env,
        best_model_save_path='./best_model/rdnn3/',
        log_path=log_dir,
        eval_freq=100_000,
        n_eval_episodes=1
    )
    model.learn(total_timesteps=total_timesteps, tb_log_name='rdnn3', callback=eval_cb)
    model.save('rdnn3_recurrentppo')
    return model

def train_recurrent_ppo_with_validation(train_env, eval_env, total_timesteps=50000, log_dir='./logs/rdnn4/'):
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=5,
        min_evals=5,
        verbose=1
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path='./best_model/rdnn4/',
        log_path=log_dir,
        eval_freq=5000,
        n_eval_episodes=5,
        callback_after_eval=stop_cb
    )
    model = RecurrentPPO(
        policy='MlpLstmPolicy',
        env=train_env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=256,
        tensorboard_log=log_dir,
    )
    model.learn(total_timesteps=total_timesteps, tb_log_name='rdnn4', callback=eval_cb)
    model.save('rdnn4_recurrentppo')
    return model

class InferenceAgent:
    """
    Loads a trained policy and provides action probability inference.
    """
    def __init__(self, model_path):
        self.model = RecurrentPPO.load(model_path)
        self.hidden_state = None

    def infer(self, window: np.ndarray) -> str:
        # build numpy episode_start mask
        episode_starts_np = np.array([self.hidden_state is None], dtype=bool)

        # init or carry over hidden state
        obs_batch = window[np.newaxis, ...]
        _, self.hidden_state = self.model.predict(
            obs_batch,
            state=self.hidden_state,
            episode_start=episode_starts_np
        )

        # prepare torch tensors
        obs_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        ep_tensor = torch.tensor(episode_starts_np, dtype=torch.float32, device=obs_tensor.device)

        # get distribution & new states
        dist, new_states = self.model.policy.get_distribution(
            obs_tensor,
            self.hidden_state,
            ep_tensor
        )
        self.hidden_state = new_states

        # detach before numpy
        probs = dist.distribution.probs.detach().cpu().numpy()[0].tolist()
        return json.dumps({"action_probs": probs})

    def benchmark(self, window: np.ndarray, n_runs: int = 100) -> float:
        obs_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
        # warm-up
        _ = self.model.policy.get_distribution(
            obs_tensor,
            self.hidden_state or np.array([True], dtype=bool),
            torch.tensor([0.0], dtype=torch.float32, device=obs_tensor.device)
        )
        start = time.time()
        for _ in range(n_runs):
            with torch.no_grad():
                _ = self.model.policy.get_distribution(
                    obs_tensor,
                    self.hidden_state,
                    torch.tensor([0.0], dtype=torch.float32, device=obs_tensor.device)
                )
        latency = (time.time() - start) / n_runs
        print(f"Average inference latency: {latency*1000:.3f} ms")
        return latency

if __name__ == '__main__':
    ticker = input('Enter stock ticker symbol (e.g., AAPL): ').strip().upper()
    try:
        data_array = fetch_ohlcv(ticker)
        print(f'Fetched {data_array.shape[0]} rows of OHLCV data for {ticker}.')
    except Exception as e:
        print(f'Error fetching data for {ticker}: {e}')
        exit(1)

    window_size = 10
    split_idx = int(len(data_array) * 0.8)
    train_data = data_array[:split_idx]
    test_data  = data_array[split_idx - window_size:]
    train_env  = TradingEnv(train_data, window_size=window_size)
    eval_env   = TradingEnv(test_data,  window_size=window_size)

    model = train_recurrent_ppo_with_validation(train_env, eval_env)

    agent = InferenceAgent('./best_model/rdnn4/best_model.zip')
    sample_window = test_data[:window_size]
    print(f'Action probabilities: {agent.infer(sample_window)}')
    agent.benchmark(sample_window, n_runs=500)
