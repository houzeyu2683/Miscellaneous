import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas
import math
import keras
import csv
import pickle
import h5py
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
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
from keras.layers import Dense, Dropout, Activation,Input
from keras import losses
from keras.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
import keras.backend as K
from keras.callbacks import History 
import pandas as pd
import tensorflow as tf
import argparse
import os

history = History() 
start_from = 50
train_epoch = 1000
scale = 10000
add_la= 25.08
add_lo = 121.55

numpy.set_printoptions(suppress=True)
parser = argparse.ArgumentParser()
parser.add_argument("--GPU",type=int, default=2)
parser.add_argument("--feature",type=int, default=6)
parser.add_argument("--Full_step",type=int,default = 60, help = 'Data full step') 
parser.add_argument("--model",type=str, default="../Saved_Model/Dropout/model.h5",help = 'Model path')
parser.add_argument("--batch_size",type=int, default=6000)
parser.add_argument("--Uncertainty_times",type=int, default=20,help = 'Model repeat n times to calculate uncertainty')
args = parser.parse_args()

################## GPU's utilization ####################################
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)


#------------------------------------------------------------------------
def calculate_error(Predict,cor_label):
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
    add_la= 25.08
    add_lo = 121.55
    radian = 0.0174532925
    
    predict_longitude= Predict[:,1]  / scale
    predict_latitude = Predict[:,0]  / scale
    true_longitude   = cor_label[:,1]  / scale
    true_latitude    = cor_label[:,0]  / scale
    
    predict_longitude = predict_longitude + add_lo
    true_longitude = true_longitude+ add_lo
    predict_latitude = predict_latitude + add_la
    true_latitude = true_latitude + add_la
    
    predict_longitude = predict_longitude* radian
    predict_latitude = predict_latitude* radian
    true_longitude = true_longitude* radian
    true_latitude = true_latitude* radian
    
    error = []
    error = np.array(error, np.float64)
    count_domain_error = 0
    sum = 0
    length = len(Predict)
    for i in range(len(Predict)):
        temp = math.sin(predict_latitude[i])*math.sin(true_latitude[i])+math.cos(true_latitude[i])*math.cos(predict_latitude[i])*math.cos(abs(predict_longitude[i]-true_longitude[i]))
        # print(temp, true_longitude/radian, true_latitude/radian, predict_longitude/radian, predict_latitude/radian)
        try:
            temp = math.acos(temp)
        except ValueError:
            length = length - 1
            error = np.append(error, 101)
            count_domain_error = count_domain_error +1
            continue
            
        distance = temp*6371.009*1000
        sum = sum + distance
        error = np.append(error, distance)
    answer = sum/length
    print("Count_domain_error is %d"%(count_domain_error))
    return answer, error
#-----------------------------------------------------------------------------------------------------------------------
def Testing(loadmodel,file_dir,file_destination,batch,times):
    for file in os.listdir(file_dir):
        dataset_test = numpy.loadtxt(f"{file_dir}/{file}", delimiter=",")
        testing_magnetic_feature = dataset_test[:,0:args.step*args.feature]
        testing_cor_label = dataset_test[:,args.step*args.feature:args.step*args.feature+2]
        step = int(testing_magnetic_feature.shape[1]/args.feature)

        if (step<args.Full_step):
            Mask_array = []
            Mask_array = np.array (Mask_array)
            Mask_array = np.append(Mask_array,np.full((len(testing_magnetic_feature)*int(args.Full_step-step)*args.feature), fill_value=special_value))
            Mask_array = Mask_array.reshape((len(testing_magnetic_feature),int(args.Full_step-step)*args.feature))
            testing_magnetic_feature = np.concatenate((testing_magnetic_feature,Mask_array),axis = 1)


        testing_magnetic_feature = numpy.reshape(testing_magnetic_feature, (testing_magnetic_feature.shape[0], args.Full_step,args.feature))    
        testPredict= loadmodel.predict(testing_magnetic_feature)

    
        testPredict = np.array(testPredict, np.float64)
        test_loss,test_loss_array = calculate_error(testPredict,testing_cor_label)
        print(f"{file}:")
        print("Test_loss is %.10f"%(test_loss))
        print("")

        Yt_hat = np.array([])
        for i in range(int(len(testing_magnetic_feature)/batch)+1):
            temp = np.array([loadmodel(testing_magnetic_feature[batch*i:batch*(i+1),:,:],training=True) for _ in range(times)])
            if i == 0:
                Yt_hat = temp
            else:
                Yt_hat = np.concatenate([Yt_hat,temp],axis=1)
        uncertainty_1 = np.var(Yt_hat[:,:,0] , axis = 0)
        uncertainty_2 = np.var(Yt_hat[:,:,1] , axis = 0)
        uncertainty_all = (uncertainty_1) + (uncertainty_2)

 #--------------------------------------------------------------------------------------------------------------------
        testPredict = testPredict/scale
        testing_cor_label = testing_cor_label/scale
        testPredict[:,1] = testPredict[:,1] + add_lo
        testPredict[:,0] = testPredict[:,0] + add_la
        testing_cor_label[:,1] = testing_cor_label[:,1] + add_lo
        testing_cor_label[:,0] = testing_cor_label[:,0] + add_la
        testPredict = pd.DataFrame(testPredict)
        testing_cor_label = pd.DataFrame(testing_cor_label)
        uncertainty_all = pd.DataFrame(uncertainty_all)
        frames = [testPredict,testing_cor_label,uncertainty_all]
        result = pd.concat(frames,axis = 1,sort = False)
        filepath = f"./{file_destination}/{file}"
        filename = f"{file_destination}/{file}"
        if os.path.isfile(filepath):
            os.remove(filename)
        result.to_csv(filename,index=False,float_format='%.20f')        
#-----------------------------------------------------------------------------------------------------------------------
loadmodel = load_model(args.model)

special_value = -1000
batch = args.batch_size
times = args.Uncertainty_times

file_dir = "1029_fixed_speed_sample"
file_destination = "1029_Miramar_result"
Testing(loadmodel,file_dir,file_destination,batch,times)

file_dir = "1030_fixed_speed_sample"
file_destination = "1030_Miramar_result"
Testing(loadmodel,file_dir,file_destination,batch,times)


file_dir = "20201208_speed_sample"
file_destination = "1208_Miramar_result"
Testing(loadmodel,file_dir,file_destination,batch,times)


