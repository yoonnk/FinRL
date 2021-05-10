import site
site.addsitedir(r"C:\Users\USER\Desktop\new\dl4seq")
site.addsitedir(r"D:\mytools\AI4Water")

from AI4Water import DualAttentionModel
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

cpath = r"TOATHER/20210510_133918/config.json"

df = pd.read_excel(r'TOATHER/data_betwwen_CIP.xlsx')
df.index = pd.date_range("20110101", periods=len(df), freq='S')


model = DualAttentionModel.from_config(cpath, data=df)

model.load_weights('weights_209_0.00007.hdf5')


def call_model(inputs, prev_inputs):

    input_dict = {'enc_input': inputs,
                  'input_y': prev_inputs,
                  'enc_first_cell_state_1': np.zeros((1, 20)),
                  'enc_first_hidden_state_1': np.zeros((1, 20)),
                  'dec_1st_cell_state': np.zeros((1, 30)),
                  'dec_1st_hidden_state': np.zeros((1, 30))
                  }

    preds = model._model.predict(x=list(input_dict.values()))
    predic = preds[0][0][0]

    return predic

if __name__ == "__main__":

    _inputs = np.random.random((1, 3, 6))
    _prev_inputs = np.random.random((1, 2, 1))

    pred = call_model(_inputs, _prev_inputs)
