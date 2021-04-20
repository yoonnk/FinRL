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

def call_model(flow_rate):
    cpath = r"C:\Users\USER\Desktop\test_dl4seq\results\20210412_132712\config.json"

    df = pd.read_excel(r'C:\Users\USER\Desktop\test_dl4seq\data\1YRDATA.xlsx')
    # df.index = pd.to_datetime(df['date'])
    # df = df.dropna(axis=0)
    df.index = pd.date_range("20110101", periods=len(df), freq='S')

    # inputs = ['유입압력']  # 'Conductivity_in', 'Conductivity_out', 'pH_in', 'Voltage'
    # outputs = ["FLUXSFX"]

    model = Model.from_config(cpath,data=df)

    #history = model.fit(data=df, indice='random')
    model.load_weights('weights_010_0.07438.hdf5')

    # preds = model.predict(use_datetime_index= True)
    preds = model.predict(st=flow_rate, en=flow_rate+2, use_datetime_index=True)
    predic = preds[1][0]

    return int(predic)

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

