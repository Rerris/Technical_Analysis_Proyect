# Required Libraries
import gymnasium
from gymnasium import spaces
import numpy as np
import pandas as pd
import ta
import matplotlib.pyplot as plt
import requests
import io

class TradingEnv(gymnasium.Env):
    """Custom trading environment incorporating technical indicators."""
    metadata = {'render.modes': ['human']}

    def __init__(self, data, initial_investment=20000):
        super(TradingEnv, self).__init__()
        self.stock_price_history = data['Close'].values
        self.rsi = data['RSI'].values
        self.macd = data['macd'].values
        self.macd_signal = data['macd_signal'].values
        self.ema_13 = data['Ema 13'].values
        self.ema_48 = data['Ema 48'].values
        self.ema_200 = data['Ema 200'].values
        self.n_step = self.stock_price_history.shape[0]
        
        # Initialize state
        self.current_step = None
        self.cash = None
        self.stock_owned = None
        self.stock_price = None
        self.history = []
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,))
        self.initial_investment = initial_investment
        self.reset()

    def step(self, action):
        assert self.action_space.contains(action)

        prev_val = self._get_val()
        self.current_step += 1
        self.stock_price = self.stock_price_history[self.current_step]
        
        # Execute trade
        self._trade(action)

        # Calculate portfolio value and reward
        current_val = self._get_val()
        reward = current_val - prev_val - 1

        # Determine if episode is complete
        done = self.current_step == self.n_step - 1

        info = {'portfolio_value': current_val}
        return self._get_obs(), reward, done, info

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_investment
        self.stock_owned = 0
        self.stock_price = self.stock_price_history[self.current_step]
        self.history = []
        return self._get_obs()

    def _get_obs(self):
        """Include stock price, technical indicators, and portfolio data in the observation space."""
        return np.array([
            self.stock_price, 
            self.rsi[self.current_step], 
            self.macd[self.current_step], 
            self.macd_signal[self.current_step], 
            self.ema_13[self.current_step], 
            self.ema_48[self.current_step], 
            self.ema_200[self.current_step], 
            self.stock_owned, 
            self.cash
        ])

    def _get_val(self):
        """Calculate the total value of the current portfolio."""
        return self.stock_owned * self.stock_price + self.cash

    def _trade(self, action):
        if action == 1:  # Buy
            max_stocks = self.cash // self.stock_price
            self.stock_owned += max_stocks
            self.cash -= self.stock_price * max_stocks
            self.history.append(('buy', max_stocks))
        elif action == 2:  # Sell
            self.cash += self.stock_owned * self.stock_price
            self.history.append(('sell', self.stock_owned))
            self.stock_owned = 0

    def render(self, mode='human', close=False):
        profit = self._get_val() - self.initial_investment
        print(f'Step: {self.current_step}')
        print(f'Stock Price: ${self.stock_price:.2f}')
        print(f'Stocks Owned: {self.stock_owned}')
        print(f'Cash: ${self.cash:.2f}')
        print(f'Profit: ${profit:.2f}')

# Example Execution
env = TradingEnv(data)
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # Random action selection
    obs, reward, done, info = env.step(action)
    if done:
        env.render()

# Plot the Portfolio Value Over Time
plt.plot([h['portfolio_value'] for h in env.history])
plt.title('Portfolio Value Over Time')
plt.xlabel('Steps')
plt.ylabel('Portfolio Value')
plt.show()
