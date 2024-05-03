import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class TradingEnv(gym.Env):

    metadata = {'render.modes': ['human']}
    
    def __init__(self, data, initial_investment=20000):
        super(TradingEnv, self).__init__()
        self.stock_price_history = data['Close'].values
        self.n_step = self.stock_price_history.shape[0]
        self.current_step = None
        self.cash = None
        self.stock_owned = None
        self.stock_price = None
        self.history = []
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(3,))
        self.reset()

    def step(self, action):
        assert self.action_space.contains(action)
        prev_val = self._get_val()
        self.current_step += 1
        self.stock_price = self.stock_price_history[self.current_step]
        self._trade(action)
        current_val = self._get_val()
        reward = current_val / prev_val - 1
        reward= (reward- (-0.02))/(0.02-(-0.02))
        reward=reward*2-1
        done = self.current_step == self.n_step - 1
        info = {'portfolio_value': current_val}
        return self._get_obs(), reward, done, info

    def reset(self):
        self.current_step = 0
        self.cash = self.initial_investment
        self.stock_owned = 0
        self.stock_price = self.stock_price_history[self.current_step]
        return self._get_obs()

    def _get_obs(self):
        return np.array([self.stock_price, ])

    def _get_val(self):
        return self.stock_owned * self.stock_price + self.cash

    def _trade(self, action):
        if action == 1:
            max_stocks = self.cash // self.stock_price
            self.stock_owned += max_stocks
            self.cash -= self.stock_price * max_stocks
        elif action == 2:
            self.cash += self.stock_owned * self.stock_price
            self.stock_owned = 0

    def render(self, mode='human', close=False):
        profit = self._get_val() - self.initial_investment
        print(f'Step: {self.current_step}')
        print(f'Stock Price: ${self.stock_price:.2f}')
        print(f'Stocks Owned: {self.stock_owned}')
        print(f'Cash: ${self.cash:.2f}')
        print(f'Profit: ${profit:.2f}')


data = pd.read_csv('path_to_your_data.csv')  # Asegúrate de tener la ruta correcta


env = TradingEnv(data)
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # Selección aleatoria de la acción
    obs, reward, done, info = env.step(action)
    if done:
        env.render()


plt.plot([h['portfolio_value'] for h in env.history])
plt.title('Portfolio Value Over Time')
plt.xlabel('Steps')
plt.ylabel('Portfolio Value')
plt.show()
