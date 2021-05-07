import site
site.addsitedir(r"C:\Users\USER\Desktop\new\dl4seq")

from dl4seq import Model
# import time
import pandas as pd
# from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
# from tensorflow import keras
np.set_printoptions(suppress=True)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

tf.compat.v1.disable_eager_execution()

cpath = r"C:\Users\USER\Desktop\test_dl4seq\results\bayes\20210506_210851\config.json"

df = pd.read_excel(r'C:\Users\USER\Desktop\test_dl4seq\data\DataforRL1.xlsx')
# df.index = pd.to_datetime(df['date'])
# df = df.dropna(axis=0)
df.index = pd.date_range("20110101", periods=len(df), freq='S')

# inputs = ['유입압력']  # 'Conductivity_in', 'Conductivity_out', 'pH_in', 'Voltage'
# outputs = ["FLUXSFX"]

model = Model.from_config(cpath, data=df)

# history = model.fit(data=df, indice='random')
model.load_weights('weights_116_0.02818.hdf5')

# def call_model(PRESSURE):
#
#     # preds = model.predict(use_datetime_index= True)
#     #PP = [PRESSURE, PRESSURE]
#     ppp = np.array(PRESSURE)
#     # preds = model.predict(use_datetime_index= True)
#     p = ppp.reshape(1, 2, 1)
#     preds = model._model.predict(x=p)
#     # preds = model.predict(st=0, en=3, use_datetime_index=True)
#     predic = preds[0][0][0]
#
#     return predic

def call_model(PP):

    # preds = model.predict(use_datetime_index= True)
    #PP = [PP for _ in range(7)]
    PP = np.array(PP)
    # preds = model.predict(use_datetime_index= True)
    p = PP.reshape(1, 9, 1)
    preds = model._model.predict(x=p)
    # preds = model.predict(st=0, en=3, use_datetime_index=True)
    predic = preds[0][0][0]

    return predic


# cpath = r"C:\Users\USER\Desktop\test_dl4seq\results\20210412_132712\config.json"
#
# df = pd.read_excel(r'C:\Users\USER\Desktop\test_dl4seq\data\1YRDATA.xlsx')
#     # df.index = pd.to_datetime(df['date'])
#     # df = df.dropna(axis=0)
# df.index = pd.date_range("20110101", periods=len(df), freq='S')
#
#     # inputs = ['유입압력']  # 'Conductivity_in', 'Conductivity_out', 'pH_in', 'Voltage'
#     # outputs = ["FLUXSFX"]
#
# model = Model.from_config(cpath,data=df)
#
#     #history = model.fit(data=df, indice='random')
# model.load_weights('weights_010_0.07438.hdf5')
#
#     # preds = model.predict(use_datetime_index= True)
# preds = model.predict(22, use_datetime_index=True)
# predic = int(preds[1][0])

# cpath = r"C:\Users\USER\Desktop\test_dl4seq\results\20210504_200201\config.json"
#
# df = pd.read_excel(r'C:\Users\USER\Desktop\test_dl4seq\data\1YRDATA.xlsx')
#     # df.index = pd.to_datetime(df['date'])
#     # df = df.dropna(axis=0)
# df.index = pd.date_range("20110101", periods=len(df), freq='S')
#
#     # inputs = ['유입압력']  # 'Conductivity_in', 'Conductivity_out', 'pH_in', 'Voltage'
#     # outputs = ["FLUXSFX"]
#
# model = Model.from_config(cpath,data=df)
#
#     #history = model.fit(data=df, indice='random')
# model.load_weights('weights_004_0.07948.hdf5')
# PP=[22,22]
# PRESSURE = np.array(PP)
#     # preds = model.predict(use_datetime_index= True)
# p = PRESSURE.reshape(1,2,1)
# preds = model._model.predict(x=p)

