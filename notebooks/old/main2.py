import yfinance as yf
from stockstats import StockDataFrame as Sdf
from env_nk_multi import BWTPEnv
import pandas as pd
import matplotlib.pyplot as plt

import gym
from stable_baselines import PPO2, DDPG, A2C, ACKTR, TD3
from stable_baselines import DDPG
from stable_baselines import A2C
from stable_baselines import SAC
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.policies import MlpPolicy

#Diable the warnings
import warnings
warnings.filterwarnings('ignore')

data_df = pd.read_excel(r'C:\Users\USER\Desktop\FinRL\data\data_betwwen_CIP.xlsx',
                        usecols=['MF_TURBIDITY', 'FEED_TEMPERATURE', 'FEED_TDS', 'FEED_FLOWRATE', 'FEED_PRESSURE', 'CIP', 'FLOWRATE'], nrows=815)

data_df=data_df.reset_index()
data_df=data_df.fillna(method='bfill')
data_clean = data_df.copy()

train = data_clean[0:600]
# the index needs to start from 0
train=train.reset_index(drop=True)

env_train = DummyVecEnv([lambda: BWTPEnv(train)])
model_ppo = PPO2('MlpPolicy', env_train, tensorboard_log="./single_stock_trading_2_tensorboard/")
model_ppo.learn(total_timesteps=50000,tb_log_name="run_aapl_ppo")

test = data_clean[600: ]
# the index needs to start from 0
test=test.reset_index(drop=True)

model = model_ppo
env_test = DummyVecEnv([lambda: BWTPEnv(test)])
obs_test = env_test.reset()
print("==============Model Prediction===========")

# for i in range(len(test.index.unique())):
#     print("testing", i, "th")
#     action, _states = model.predict(obs_test)
#     obs_test, rewards, dones, info = env_test.step(action)
#     env_test.render()

