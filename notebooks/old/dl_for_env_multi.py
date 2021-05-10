import site
site.addsitedir(r"C:\Users\USER\Desktop\new\dl4seq")

from dl4seq import DualAttentionModel
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

cpath = r"C:\Users\USER\Desktop\test_dl4seq\results\new_for_RL2\20210510_133918\config.json"

df = pd.read_excel(r'C:\Users\USER\Desktop\test_dl4seq\data\data_betwwen_CIP.xlsx')
df.index = pd.date_range("20110101", periods=len(df), freq='S')


model = DualAttentionModel.from_config(cpath, data=df)

model.load_weights('weights_209_0.00007.hdf5')


def call_model(PP):

    PP = np.array(PP)
    p = PP.reshape(1, 3, 1)
    preds = DualAttentionModel._model.predict(x=p)
    predic = preds[0][0][0]

    return predic