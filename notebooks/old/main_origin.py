
import yfinance as yf
from stockstats import StockDataFrame as Sdf
from environment import SingleStockEnv
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

def get_DRL_sharpe():
    df_total_value = pd.read_csv('account_value.csv', index_col=0)
    df_total_value.columns = ['account_value']
    df_total_value['daily_return'] = df_total_value.pct_change(1)
    sharpe = (252 ** 0.5) * df_total_value['daily_return'].mean() / \
             df_total_value['daily_return'].std()

    annual_return = ((df_total_value['daily_return'].mean() + 1) ** 252 - 1) * 100
    print("annual return: ", annual_return)
    print("sharpe ratio: ", sharpe)
    return df_total_value

def get_buy_and_hold_sharpe(test):
    test['daily_return']=test['adjcp'].pct_change(1)
    sharpe = (252**0.5)*test['daily_return'].mean()/ \
    test['daily_return'].std()
    annual_return = ((test['daily_return'].mean()+1)**252-1)*100
    print("annual return: ", annual_return)

    print("sharpe ratio: ", sharpe)
    #return sharpe


# Download and save the data in a pandas DataFrame:
data_df = yf.download("AAPL", start="2009-01-01", end="2020-10-23")
# data_df.shape
# data_df.head()
# data_df.columns

# reset the index, we want to use numbers instead of dates
data_df=data_df.reset_index()

# convert the column names to standardized names
data_df.columns = ['datadate','open','high','low','close','adjcp','volume']


# save the data to a csv file in your current folder
data_df.to_csv('AAPL_2009_2020.csv')

# check missing data
# data_df.isnull().values.any()

# calculate technical indicators like MACD
stock = Sdf.retype(data_df.copy())
# we need to use adjusted close price instead of close price
stock['close'] = stock['adjcp']
data_df['macd'] = stock['macd']

# check missing data
# data_df.isnull().values.any()
# data_df.head()
# if missing data is true,
# data_df=data_df.fillna(method='bfill')

# Note that I always use a copy of the original data to try it track step by step.
data_clean = data_df.copy()
# data_clean.head()
# data_clean.tail()


train = data_clean[(data_clean.datadate>='2009-01-01') & (data_clean.datadate<'2019-01-01')]
# the index needs to start from 0
train=train.reset_index(drop=True)
# train.head()

#tensorboard --logdir ./single_stock_tensorboard/
env_train = DummyVecEnv([lambda: SingleStockEnv(train)])
model_ppo = PPO2('MlpPolicy', env_train, tensorboard_log="./single_stock_trading_2_tensorboard/")
model_ppo.learn(total_timesteps=100000,tb_log_name="run_aapl_ppo")
#model.save('AAPL_ppo_100k')

test = data_clean[(data_clean.datadate>='2019-01-01') ]
# the index needs to start from 0
test=test.reset_index(drop=True)

model = model_ppo
env_test = DummyVecEnv([lambda: SingleStockEnv(test)])
obs_test = env_test.reset()
print("==============Model Prediction===========")

for i in range(len(test.index.unique())):
    action, _states = model.predict(obs_test)
    obs_test, rewards, dones, info = env_test.step(action)
    env_test.render()

df_total_value = get_DRL_sharpe()
get_buy_and_hold_sharpe(test)
DRL_cumulative_return = (df_total_value.account_value.pct_change(1)+1).cumprod()-1
buy_and_hold_cumulative_return = (test.adjcp.pct_change(1)+1).cumprod()-1


# matplotlib inline
fig, ax = plt.subplots(figsize=(12, 8))

plt.close()
plt.plot(test.datadate, DRL_cumulative_return, color='red',label = "DRL")
plt.plot(test.datadate, buy_and_hold_cumulative_return, label = "Buy & Hold")
plt.title("Cumulative Return for AAPL with Transaction Cost",size= 18)
plt.legend()
plt.rc('legend',fontsize=15)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.savefig('test.png')
plt.show()
# plt.imshow(test.datadate, DRL_cumulative_return)







