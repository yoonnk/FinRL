import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
from sklearn.preprocessing import MinMaxScaler

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dl_for_env_multi import call_model

# Global variables
PLANT_DIM = 1
EFF_PUMP = 0.9
EFF_ERD = 0.8
# FLOW_FEED = 1000
lookback_size = 3


REWARD_SCALING = 1e-2

class BWTPEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, day=2):
        # super(StockEnv, self).__init__()
        # date increment
        self.day = day
        self.df = df
        # action_space normalization and the shape is PLANT_DIM
        self.action_space = spaces.Box(low=-1, high=1, shape=(PLANT_DIM,))
        # Shape = 4: [Current Balance]+[prices]+[owned shares] +[macd]
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(8,))
        # load data from a pandas dataframe
        self.data = self.df.loc[self.day, :]
        # termination
        self.terminal = False
        # save the total number of trades
        self.trades = 0
        # initalize state
        self.state = [0.0] + \
                     [self.data.MF_TURBIDITY] + \
                     [self.data.FEED_TEMPERATURE] + \
                     [self.data.FEED_TDS] + \
                     [self.data.FEED_FLOWRATE] + \
                     [self.data.FEED_PRESSURE] + \
                     [self.data.CIP] + \
                     [self.data.FLOWRATE]

        # initialize reward and cost
        self.reward = 0
        self.cost = 0
        self.energy_difference=0
        self.total_energy_difference = 0
        self.total_reward = 0
        self.actual_flowrate = 0
        self.actual_pressure = 0
        self.actual_energy = 0
        self.optimize_energy = 0
        # self.total_actual_energy = 0
        self.total_optimize_energy = 0
        self.rewardsum = 0
        # memorize the total value, total rewards
        self.energy_memory = []
        self.rewards_memory = []
        self.rewardsum_memory = []
        self.energy_difference_memory = []
        self.total_energy_difference_memory = []
        self.action_container = self.df['FEED_PRESSURE'][0:3].to_list()
        self.total_actual_energy = 0
        # self.total_actual_energy = 1502.87059974546
        self.action_memory = []
        self.memory = []

    def change_pressure(self, action):

        self.action_memory.append(action)
        self.action_container = np.roll(self.action_container, -1)
        self.action_container[-1] = action

        self.actual_flowrate = self.state[7]
        self.actual_pressure = self.state[5]

        day = self.day
        st = day-1
        en = st + (lookback_size-1)
        _inputs=  self.df.loc[st:en, :]
        _inputs['FEED_PRESSURE'] = self.action_container
        _flow_rate = _inputs.pop('FLOWRATE')
        _inputs.pop('index')

        prediction = call_model(_inputs.values.reshape(1, lookback_size, 6), _flow_rate.values[0:-1].reshape(1, lookback_size-1, 1))

        self.state[7] = prediction * 0.5

        self.state[5] = float(action)
        # energy consuption calculation

        self.actual_energy =\
            ((self.state[4] * self.actual_pressure - EFF_PUMP * (self.actual_pressure - 3)) / 36 / EFF_ERD / self.actual_flowrate) + ((self.actual_pressure - EFF_PUMP * (self.actual_pressure - 3)) * (self.state[4] - self.actual_flowrate) / EFF_ERD / 36 /self.actual_flowrate)

        self.state[0] = ((self.state[4] * self.state[5] - EFF_PUMP * (self.state[5] - 3)) / 36 / EFF_ERD / self.state[
            7]) + ((self.state[5] - EFF_PUMP * (self.state[5] - 3)) * (self.state[4] - self.state[7]) / EFF_ERD / 36 /
                   self.state[7])

        self.memory.append([self.state[0], self.state[5], self.state[7], self.actual_energy])
        #if 0.1 < self.state[0] < 0.6:  # theoretically the range of SEC should be between 0.1 and 0.6

        self.state[5] = action  # pressure
        self.optimize_energy = self.state[0]
        self.total_optimize_energy += self.optimize_energy
        self.total_actual_energy += self.actual_energy

        self.energy_difference = self.actual_energy- self.optimize_energy
        # self.total_energy_difference = self.total_self.actual_energy - self.total_optimize_energy

        # update held shares
        #self.state[index + PLANT_DIM + 1] += min(available_amount, action)
        # # update transaction costs
        self.cost += self.state[0]*10
        self.trades += 1

        # else:
        #     self.action_container[-1] = self.state[3]
        #     self.state[0] = self.actual_energy
        #     self.state[7] = self.actual_flowrate
        #     self.state[5] = self.actual_pressure
        #     self.optimize_energy = self.actual_energy
        #     self.total_optimize_energy += self.optimize_energy
        #     self.total_actual_energy += self.actual_energy
        #
        #     self.energy_difference = 0
        #     # self.total_energy_difference = self.total_actual_energy - self.total_optimize_energy
        #
        #     # update held shares
        #     #self.state[index + PLANT_DIM + 1] += min(available_amount, action)
        #     # # update transaction costs
        #     self.cost += self.state[0]*10




    def step(self, actions):
        # actions is a list of floats of length=1
        self.terminal = self.day >= len(self.df.index.unique()) - 1

        if self.terminal:
            #plt.plot(self.energy_memory, 'r')
            #plt.savefig('account_value.png')
            #plt.close()

            # pd.DataFrame(np.array(self.action_memory)).to_csv('action_memory.csv')

            end_energy = self.state[0]
            self.total_energy_difference = self.total_actual_energy - self.total_optimize_energy
            print("previous_total_energy:{}".format(self.energy_memory[0]))
            print("end_energy:{}".format(end_energy))
            print("total_actual_energy:{}".format(self.total_actual_energy))
            print("total_optimize_energy:{}".format(self.total_optimize_energy))
            print("total_energy_difference:{}".format(self.total_energy_difference))

            df_total_value = pd.DataFrame( self.rewards_memory)
            df_total_value.to_csv('reward_memory_3.csv')
            df_total_value = pd.DataFrame(self.energy_difference_memory)
            df_total_value.to_csv('energy_difference_memory_3.csv')
            # df_total_value = pd.DataFrame(self.total_energy_difference)
            # df_total_value.to_csv('total_energy_difference_2.csv')
            df_total_value = pd.DataFrame(self.energy_memory)
            df_total_value.to_csv('energy_memory_3.csv')
            df_total_value = pd.DataFrame(self.total_energy_difference_memory)
            df_total_value.to_csv('total_energy_difference_memory_3.csv')
            df_total_value = pd.DataFrame(self.action_memory)
            df_total_value.to_csv('action_memory_3.csv')
            print("reward:{}".format(self.reward))
            print("rewardsum:{}".format(self.rewardsum))
            print("total_cost: ", self.cost)
            print("total trades: ", self.trades)


            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.to_csv('rewards_3.csv')
            # df_rewards = pd.DataFrame(self.rewardsum_memory)
            # df_rewards.to_csv('rewardsum_2.csv')

            df = pd.DataFrame(self.memory, columns=['pred_energy', 'action', 'prediction', 'actual_engergy'])
            df.to_csv('memory.csv')

            return self.state, self.reward, self.terminal, {}

        else:
            actions = actions *5 + 10 # * HMAX_NORMALIZE
            self.change_pressure(actions)

            # update data, walk a step s'
            self.day += 1
            self.data = self.df.loc[self.day, :]

            self.state = [0.0] + \
                         [self.data.MF_TURBIDITY] + \
                         [self.data.FEED_TEMPERATURE] + \
                         [self.data.FEED_TDS] + \
                         [self.data.FEED_FLOWRATE] + \
                         [self.data.FEED_PRESSURE] + \
                         [self.data.CIP] + \
                         [self.data.FLOWRATE]

            end_energy = self.state[0]

            self.reward = self.energy_difference # begin_total_energy - end_total_energy
            self.rewardsum +=self.reward
            # self.reward = self.reward * 10
            self.rewards_memory.append(self.reward)
            self.rewardsum_memory.append(self.rewardsum)
            # self.reward = self.reward # * REWARD_SCALING
            self.energy_memory.append(end_energy)
            self.energy_difference_memory.append(self.energy_difference)

        self.rewardsum_memory.append(self.rewardsum)
        df_rewards = pd.DataFrame(self.rewardsum_memory)
        df_rewards.to_csv('rewardsum_3.csv')
        return self.state,self.reward, self.terminal, {}

    def reset(self):
        #self.energy_memory = [INITIAL_ENERGY]
        self.day = 2
        self.data = self.df.loc[self.day, :]
        self.cost = 0
        self.trades = 0
        self.rewardsum = 0
        self.total_actual_energy = 0.0
        self.total_optimize_energy = 0.0
        self.total_energy_difference = 0.0
        self.terminal = False
        self.rewards_memory = []
        # initiate state
        self.state = [0.0] + \
                     [self.data.MF_TURBIDITY] + \
                     [self.data.FEED_TEMPERATURE] + \
                     [self.data.FEED_TDS] + \
                     [self.data.FEED_FLOWRATE] + \
                     [self.data.FEED_PRESSURE] + \
                     [self.data.CIP] + \
                     [self.data.FLOWRATE]
        return self.state

    def render(self, mode='human'):
        return self.state

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]