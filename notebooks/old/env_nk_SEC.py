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
INITIAL_ENERGY = 1000
PLANT_DIM = 1
EFF_PUMP = 0.95
EFF_ERD = 0.8
FLOW_FEED = 800

# transaction fee: 1/1000 reasonable percentage
# TRANSACTION_FEE_PERCENT = 0.001

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
        self.state = [INITIAL_ENERGY] + \
                     [self.data.flow] + \
                     [0] * PLANT_DIM + \
                     [self.data.pressure]
        # initialize reward and cost
        self.reward = 0
        self.cost = 0

        # memorize the total value, total rewards
        self.energy_memory = [INITIAL_ENERGY]
        self.rewards_memory = []
        self.action_container = [float(self.df['pressure'].mean()) for _ in range(2)]

    #
    # def _increase_pressure(self, index, action):
    #
    #     if self.state[index + PLANT_DIM + 1] > 0:
    #         self.state[1] = call_model(action)
    #         self.state[3] =  action
    #         # energy consumption calculation
    #         self.state[0] += \
    #             (((self.state[1]*self.state[3])/EFF_PUMP)-(((self.state[3]-EFF_ERD*(self.state[3]-3))*(FLOW_FEED-self.state[1]))/EFF_PUMP))/self.state[3]
    #         self.state[index + PLANT_DIM + 1] -= min(abs(action), self.state[index + PLANT_DIM + 1])
    #         self.trades += 1
    #         # # update transaction costs
    #         self.cost += self.state[0]*1000
    #
    #     else:
    #         pass
    #
    # def _decrease_pressure(self, index, action):
    #
    #     available_amount = self.state[0] // self.state[index + 1]
    #     # update balance
    #     self.state[1] = call_model(self.action_container)
    #     self.state[3] = action
    #     # energy consuption calculation
    #     self.state[0] += \
    #         (((self.state[1]*self.state[3])/EFF_PUMP)-(((self.state[3]-EFF_ERD*(self.state[3]-3))*(FLOW_FEED-self.state[1]))/EFF_PUMP))/self.state[3]
    #     # update held shares
    #     self.state[index + PLANT_DIM + 1] += min(available_amount, action)
    #     # # update transaction costs
    #     self.cost += self.state[0]*1000
    #     self.trades += 1

    def change_pressure(self, action):
        #available_amount = self.state[0] // self.state[index + 1]
        # update balance
        self.state[1] = call_model(self.action_container)
        self.state[3] = action
        # energy consuption calculation
        self.state[0] += \
            (((self.state[1]*self.state[3])/EFF_PUMP) \
             -(((self.state[3]-EFF_ERD*(self.state[3]-3))*(FLOW_FEED-self.state[1]))/EFF_PUMP))/self.state[3]
        # update held shares
        #self.state[index + PLANT_DIM + 1] += min(available_amount, action)
        # # update transaction costs
        self.cost += self.state[0]*1000
        self.trades += 1


    def step(self, actions):
        # actions is a list of floats of length=1
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            plt.plot(self.energy_memory, 'r')
            plt.savefig('account_value.png')
            plt.close()

            end_total_energy = self.state[0] + \
                              sum(np.array(self.state[1:(PLANT_DIM + 1)]) * np.array(
                                  self.state[(PLANT_DIM + 1):(PLANT_DIM * 2 + 1)]))
            print("previous_total_energy:{}".format(self.energy_memory[0]))
            print("end_total_energy:{}".format(end_total_energy))

            df_total_value = pd.DataFrame(self.energy_memory)
            df_total_value.to_csv('account_value.csv')
            print("total_reward:{}".format(self.state[0] + sum(np.array(self.state[1:(PLANT_DIM + 1)]) * np.array(
                self.state[(PLANT_DIM + 1):(PLANT_DIM * 2 + 1)])) - INITIAL_ENERGY))
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
            # calculate begining total energy
            # begin_total_energy = self.state[0] + \
            #                     sum(np.array(self.state[1:(PLANT_DIM + 1)]) * np.array(
            #                         self.state[(PLANT_DIM + 1):(PLANT_DIM * 2 + 1)]))
            begin_total_energy = INITIAL_ENERGY

            # perform buy or sell action
            #argsort_actions = np.argsort(actions)
            #decrease_index = argsort_actions[:np.where(actions < 0)[0].shape[0]]
            #increase_index = argsort_actions[::-1][:np.where(actions > 0)[0].shape[0]]


            self.action_container = np.roll(self.action_container, -1)
            self.action_container[-1] = actions

            # for index in increase_index:
            #     # print('take sell action'.format(actions[index]))
            #     self._increase_pressure(index, actions[index])
            #
            # for index in decrease_index:
            #     # print('take buy action: {}'.format(actions[index]))
            #     self._decrease_pressure(index, actions[index])

            self.change_pressure(actions[0])

            # update data, walk a step s'
            self.day += 1
            self.data = self.df.loc[self.day, :]
            # load next state
            self.state = [self.state[0]] + \
                         [self.data.flow] + \
                         list(self.state[(PLANT_DIM + 1):(PLANT_DIM * 2 + 1)]) + \
                         [self.data.pressure]

            # calculate the end total energy
            # end_total_energy = self.state[0] + \
            #                   sum(np.array(self.state[1:(PLANT_DIM + 1)]) * np.array(
            #                       self.state[(PLANT_DIM + 1):(PLANT_DIM * 2 + 1)]))
            end_total_energy = self.state[0]

            self.reward = begin_total_energy - end_total_energy
            self.rewards_memory.append(self.reward)
            # self.reward = self.reward * REWARD_SCALING
            self.energy_memory.append(end_total_energy)

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.energy_memory = [INITIAL_ENERGY]
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.cost = 0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        # initiate state
        self.state = [INITIAL_ENERGY] + \
                     [self.data.flow] + \
                     [0] * PLANT_DIM + \
                     [self.data.pressure]
        return self.state

    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]