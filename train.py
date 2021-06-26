import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
import keras
import csv
import pickle
import h5py
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Masking
from keras.layers import Dropout
from keras.layers import Bidirectional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib
from keras import optimizers
import keras.backend as K
from keras.models import load_model
from numpy.linalg import inv
from keras.callbacks import ModelCheckpoint
import keras
import keras.utils
from keras.layers import Dense, Dropout, Activation
from keras import losses
from keras.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
import keras.backend as K
from keras.callbacks import History
import pandas as pd
import tensorflow as tf
import argparse
import os
import random


parser = argparse.ArgumentParser()
parser.add_argument("--GPU", type=int, default=1)
parser.add_argument("--epoch", type=int, default=3000)
parser.add_argument("--feature", type=int, default=6)
parser.add_argument("--step", type=int, default=60)
parser.add_argument("--bs", type=int, default=1024, help='batch size')
parser.add_argument("--model_name", type=str, default="Suture0312", help='Create ./Saved_Model/model_name')
args = parser.parse_args()
# =========================================================================
seed_value = 1
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
numpy.random.seed(seed_value)
tf.compat.v1.set_random_seed(seed_value)
# ========================GPU's utilization ========================
fraction = 0.8
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = fraction
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)
# ======================== Fix parameters  and Settings ========================
history = History()
scale = 10000
add_la = 25.08
add_lo = 121.55
numpy.set_printoptions(suppress=True)
# ------------------------------------------------------------------------


def calculate_error(Predict, cor_label):
    """
    Input:
    param Predict : Predict's lon and lat
    param cor_label: Ground Truth lon and lat
    Return:
    param answer : Averge distance error 
    param error : An array for every datapoint distance error
    """
    Predict = np.array(Predict, np.float64)
    cor_label = np.array(cor_label, np.float64)
    scale = 10000
    add_la = 25.08
    add_lo = 121.55
    radian = 0.0174532925

    predict_longitude = Predict[:, 1] / scale
    predict_latitude = Predict[:, 0] / scale
    true_longitude = cor_label[:, 1] / scale
    true_latitude = cor_label[:, 0] / scale

    predict_longitude = predict_longitude + add_lo
    true_longitude = true_longitude + add_lo
    predict_latitude = predict_latitude + add_la
    true_latitude = true_latitude + add_la

    predict_longitude = predict_longitude * radian
    predict_latitude = predict_latitude * radian
    true_longitude = true_longitude * radian
    true_latitude = true_latitude * radian

    error = []
    error = np.array(error, np.float64)
    count_domain_error = 0
    sum = 0
    length = len(Predict)
    for i in range(len(Predict)):
        temp = math.sin(predict_latitude[i])*math.sin(true_latitude[i])+math.cos(true_latitude[i])*math.cos(
            predict_latitude[i])*math.cos(abs(predict_longitude[i]-true_longitude[i]))
        try:
            temp = math.acos(temp)
        except ValueError:
            length = length - 1
            error = np.append(error, 101)
            count_domain_error = count_domain_error + 1
            continue

        distance = temp*6371.009*1000
        sum = sum + distance
        error = np.append(error, distance)
    answer = sum/length
    return answer, error


# -----------------Load training data and shuffle it --------------------------------------------------------
dataset_train = numpy.loadtxt("./Suture0311.csv", delimiter=",")
dataset_train = np.array(dataset_train, np.float64)
numpy.random.shuffle(dataset_train)
print(f"dataset_train shape:{dataset_train.shape}")
# ------------------Want to have Feature and Label-------------------------------------------------------
train_magnetic_feature = dataset_train[:, 0:args.step*args.feature]
train_cor_label = dataset_train[:, args.step *args.feature:args.step*args.feature+2]
train_magnetic_feature = numpy.reshape(train_magnetic_feature, (train_magnetic_feature.shape[0], args.step, args.feature))
# ------------------Make dir--------------------------------------------------------------------------
os.makedirs("Saved_Model/"+args.model_name, exist_ok=True)
dirpath = "./Saved_Model/%s" % (args.model_name)
if os.path.isdir(dirpath):
    for file in os.listdir(dirpath):
        filename = "./Saved_Model/%s/%s" % (args.model_name, file)
        os.remove(filename)
# ----------------Create Network ---------------------------------------------------------------
special_value = -1000
model = Sequential()
model.add(Masking(mask_value=special_value,input_shape=(args.step, args.feature)))
model.add(LSTM(80, return_sequences=True))
model.add(LSTM(120, return_sequences=True))
model.add(LSTM(120))
model.add(Dropout(0.2, input_shape=(120,)))
model.add(Dense(80, activation='relu'))
model.add(keras.layers.BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2, input_shape=(80,)))
model.add(keras.layers.BatchNormalization())
model.add(Dense(30, activation='relu'))
model.add(Dropout(0.2, input_shape=(64,)))
model.add(keras.layers.BatchNormalization())
model.add(Dense(2,))
save_dir = f"./Saved_Model/{args.model_name}"
filepath = "model.h5"
checkpoint = ModelCheckpoint(os.path.join(save_dir, filepath), monitor='loss', verbose=0, save_best_only=True, mode='min')
Earlystop = tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', patience=80)
Learningrate = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=30, verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
callbacks_list = [checkpoint, history, Learningrate, Earlystop]
opt = optimizers.Adam(learning_rate=1e-3)
model.compile(loss='mse', optimizer=opt)
model.fit(train_magnetic_feature, train_cor_label, epochs=args.epoch,batch_size=args.bs, callbacks=callbacks_list, verbose=1,  validation_split=0.2)
# -------------------------------------------------------------------------------------------------------------------------
