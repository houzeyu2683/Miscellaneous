import csv
import numpy
import pandas as pd
import os
import random
import numpy as np
import math
import random

filepath = "./Suture0311.csv"
filename = "Suture0311.csv"
if os.path.isfile(filepath):
    os.remove(filename)
random.seed(1234)
add_la = 25.08
add_lo = 121.55
scale=10000
step = 60
choose_start = [0,0,0,0,0]
target_distance = [1, 1.5, 2, 2.5, 3, 3.5, 4.5, 5, 6]
tolerance = 0.05
speed_min = 0.30
speed_max = 7
interval = 0.5

print(f"Now processing step {step}")

Dataset_dir = "./train"
file = os.listdir(Dataset_dir)[0]

'try to understand how processing working'
file_ = f"{Dataset_dir}/{file}"
filepath = filepath
filename = filename
add_la = add_la
add_lo = add_lo
scale = scale
step = step
choose_start = choose_start
target_distance_array = target_distance
tolerance_low = tolerance
tolerance_high = tolerance
speed_max = speed_max
speed_min = speed_min
interval = interval

df = pd.read_csv(f"{Dataset_dir}/{file}")
df = df[["MagX", "MagY", "MagZ", "Latitude", "Longitude"]]
raw_dataset_train = df.to_numpy()
raw_dataset_train = np.array(raw_dataset_train, np.float64)
raw_dataset_train.shape

'try to understand how sample_sequence working'
raw_dataset_train
filename
add_la
add_lo
scale
step
choose_start
target_distance_array
tolerance_low
tolerance_high
speed_max
speed_min
interval

special_value = -1000
for target_distance in target_distance_array:
    target_distance = target_distance_array[0]
    print(f"=======================\nNow processing distance {target_distance} \n=======================")
    for s in choose_start:
        s = choose_start[0]
        for case in range(3):
            case = 0
            dataset_train = fixed_distance(raw_dataset_train[s:, :],target_distance, tolerance_low, tolerance_high, speed_max, speed_min, interval, case)

            #================================Want to evaluate second differential================================
            second_dataset_train = np.zeros((dataset_train.shape[0],dataset_train.shape[1]))
            for i in range(len(second_dataset_train)):
                if i==0:
                    continue
                else:
                    second_dataset_train[i] = dataset_train[i] - dataset_train[i-1]
            
            #---------------Setting some paramaters-------------------------------------
            count = 0
            f = open(filename,'a')
            w = csv.writer(f,delimiter=',',lineterminator='\r\n')

            total = dataset_train.shape[0]
            sample_number= total-step
            if sample_number < 0:
                continue
            li = [i for i in range((step-1),total)]
            res = random.sample(li,sample_number)
            #---------------Want to calculate feature-------------------------------------
            #---------------data for full_step (60) ----------------------------------------
            #---------------data1 for half_step (30) ----------------------------------------
            data=[]
            data = np.array(data, np.float64)
            data1=[]
            data1 = np.array(data, np.float64)
            for positions in res:
                #-----------------Positive direction--------------------
                #positions is the final data points in this path
                for cnt, i in enumerate(range((positions-(step-1)),positions+1)):
                    if cnt >= step/2:
                        data1 = np.append(data1,dataset_train[i][0]-dataset_train[i-1][0])
                        data1 = np.append(data1,dataset_train[i][1]-dataset_train[i-1][1])
                        data1 = np.append(data1,dataset_train[i][2]-dataset_train[i-1][2])
                        data1 = np.append(data1,second_dataset_train[i][0]-second_dataset_train[i-1][0])
                        data1 = np.append(data1,second_dataset_train[i][1]-second_dataset_train[i-1][1])
                        data1 = np.append(data1,second_dataset_train[i][2]-second_dataset_train[i-1][2])
                        
                    data = np.append(data,dataset_train[i][0]-dataset_train[i-1][0])
                    data = np.append(data,dataset_train[i][1]-dataset_train[i-1][1])
                    data = np.append(data,dataset_train[i][2]-dataset_train[i-1][2])
                    data = np.append(data,second_dataset_train[i][0]-second_dataset_train[i-1][0])
                    data = np.append(data,second_dataset_train[i][1]-second_dataset_train[i-1][1])
                    data = np.append(data,second_dataset_train[i][2]-second_dataset_train[i-1][2])
                    if i == positions:
                        data1 = np.append(data1,np.full((int(step/2)*6), fill_value=special_value))         
                        data1 = np.append(data1,(dataset_train[i][3]-add_la)*scale)
                        data1 = np.append(data1,(dataset_train[i][4]-add_lo)*scale)
                        data = np.append(data,(dataset_train[i][3]-add_la)*scale)
                        data = np.append(data,(dataset_train[i][4]-add_lo)*scale)
                w.writerow(data)
                w.writerow(data1)
                data=[]
                data = np.array(data, np.float64)
                data1=[]
                data1 = np.array(data, np.float64)            


