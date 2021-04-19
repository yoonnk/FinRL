import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dl_for_env import call_model

# Global variables
HMAX_NORMALIZE = 20
INITIAL_ACCOUNT_BALANCE = 0
PLANT_DIM = 1

# transaction fee: 1/1000 reasonable percentage
TRANSACTION_FEE_PERCENT = 0.001

# REWARD_SCALING = 1e-3

class BWTPEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, day=0):
        # super(StockEnv, self).__init__()
        # date increment
        self.day = day
        self.df = df
        # action_space normalization and the shape is PLANT_DIM
        self.action_space = spaces.Box(low=-1, high=1, shape=(PLANT_DIM,))
        # Shape = 4: [Current Balance]+[prices]+[owned shares] +[macd]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4,))
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        # termination
        self.terminal = False
        # save the total number of trades
        self.trades = 0
        # initalize state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                     [self.data.flow] + \
                     [0] * PLANT_DIM + \
                     [self.data.pressure]
        # initialize reward and cost
        self.reward = 0
        self.cost = 0

        # memorize the total value, total rewards
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.rewards_memory = []


    def _increase_pressure(self, index, action):
        # perform sell action based on the sign of the action
        if self.state[index + PLANT_DIM + 1] > 0:
            # update balance
            # self.state[0] += \
            #     self.state[index + 1] * min(abs(action), self.state[index + PLANT_DIM + 1]) * \
            #     (1 - TRANSACTION_FEE_PERCENT)
            self.state[0] += \
                self.state[index + 1] * min(abs(action), self.state[index + PLANT_DIM + 1]) * \
                (1 - TRANSACTION_FEE_PERCENT)

            # update held shares
            self.state[index + PLANT_DIM + 1] -= min(abs(action), self.state[index + PLANT_DIM + 1])
            # update transaction costs
            self.cost += self.state[index + 1] * min(abs(action), self.state[index + PLANT_DIM + 1]) * \
                         TRANSACTION_FEE_PERCENT
            self.trades += 1
        else:
            pass

    def _decrease_pressure(self, index, action):
        # perform buy action based on the sign of the action
        available_amount = self.state[0] // self.state[index + 1]
        # update balance
        self.state[0] -= self.state[index + 1] * min(available_amount, action) * \
                         (1 + TRANSACTION_FEE_PERCENT)
        # update held shares
        self.state[index + PLANT_DIM + 1] += min(available_amount, action)
        # update transaction costs
        self.cost += self.state[index + 1] * min(available_amount, action) * \
                     TRANSACTION_FEE_PERCENT
        self.trades += 1


    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            plt.plot(self.asset_memory, 'r')
            plt.savefig('account_value.png')
            plt.close()

            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(PLANT_DIM + 1)]) * np.array(
                                  self.state[(PLANT_DIM + 1):(PLANT_DIM * 2 + 1)]))
            print("previous_total_asset:{}".format(self.asset_memory[0]))
            print("end_total_asset:{}".format(end_total_asset))

            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.to_csv('account_value.csv')
            print("total_reward:{}".format(self.state[0] + sum(np.array(self.state[1:(PLANT_DIM + 1)]) * np.array(
                self.state[(PLANT_DIM + 1):(PLANT_DIM * 2 + 1)])) - INITIAL_ACCOUNT_BALANCE))
            print("total_cost: ", self.cost)
            print("total trades: ", self.trades)

            df_total_value.columns = ['account_value']
            df_total_value['daily_return'] = df_total_value.pct_change(1)

            if df_total_value['daily_return'].std() != 0:
                sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / \
                         df_total_value['daily_return'].std()
                print("Sharpe: ", sharpe)
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.to_csv('account_rewards.csv')
            return self.state, self.reward, self.terminal, {}

        else:

            # actions are the shares we need to buy, hold, or sell
            actions = actions * HMAX_NORMALIZE
            # calculate begining total asset
            begin_total_asset = self.state[0] + \
                                sum(np.array(self.state[1:(PLANT_DIM + 1)]) * np.array(
                                    self.state[(PLANT_DIM + 1):(PLANT_DIM * 2 + 1)]))

            # perform buy or sell action
            argsort_actions = np.argsort(actions)
            increase_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            decrease_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]

            for index in increase_index:
                # print('take sell action'.format(actions[index]))
                self._increase_pressure(index, actions[index])

            for index in decrease_index:
                # print('take buy action: {}'.format(actions[index]))
                self._decrease_pressure(index, actions[index])
            # update data, walk a step s'
            self.day += 1
            self.data = self.df.loc[self.day, :]
            # load next state
            self.state = [self.state[0]] + \
                         [self.data.flow] + \
                         list(self.state[(PLANT_DIM + 1):(PLANT_DIM * 2 + 1)]) + \
                         [self.data.pressure]

            # calculate the end total asset
            end_total_asset = self.state[0] + \
                              sum(np.array(self.state[1:(PLANT_DIM + 1)]) * np.array(
                                  self.state[(PLANT_DIM + 1):(PLANT_DIM * 2 + 1)]))
            self.reward = end_total_asset - begin_total_asset
            self.rewards_memory.append(self.reward)
            # self.reward = self.reward * REWARD_SCALING
            self.asset_memory.append(end_total_asset)

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.asset_memory = [INITIAL_ACCOUNT_BALANCE]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        # initiate state
        self.state = [INITIAL_ACCOUNT_BALANCE] + \
                     [self.data.flow] + \
                     [0] * PLANT_DIM + \
                     [self.data.pressure]
        return self.state

    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]