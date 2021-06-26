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


"""
Define the input of "fixed_distance" function.
"""
raw_dataset_train = np.array(pd.read_csv('train/' + file)[["MagX", "MagY", "MagZ", "Latitude", "Longitude"]], np.float64)
target_distance = 1.0  # [1, 1.5, 2, 2.5, 3, 3.5 , 4, 4.5, 5, 5.5, 6]
tolerance_low = 0.5
tolerance_high = 0.5
speed_max = 7
speed_min = 0.3
interval = 0.5
case = 2 # [0, 1, 2]
def fixed_distance(raw_dataset_train, target_distance, tolerance_low, tolerance_high, speed_max, speed_min, interval, case):
    """
    Input:
      param raw_dataset_train : Raw dataset
      param target_distance : Want to find next point which distance is target_distance
      param tolerance : Because we can not find the perfect target_distance , so it need some tolerance 
      param speed_max : Upperbound of the speed , avoid driving too fast
      param speed_min : Lowerbound of the speed , avoid driving too slow
      param interval :  The allowed speed change in each time
      param case : Has three case  , "Only can accelerate" "Only can decelerate" "Both"
    Return :
      param d : Generate dataset ( one path )
    """
    #---------------------Want to choose distance between target_distance-tolerance and target_distance+tolerance
    d = [raw_dataset_train[0]]
    d = np.array(d, np.float64)
    
    
    start = raw_dataset_train[0]
    max_distance = 0  # not use, I don't known why?
    speed = target_distance
    if case == 0:
        upper = interval
        bottom = 0
    elif case == 1:
        upper = 0
        bottom = -1 * interval
    else:
        upper = interval
        bottom = -1 * interval

    for i in range(1,raw_dataset_train.shape[0]): 
        distance = calculate_distance(start[3], start[4], raw_dataset_train[i][3], raw_dataset_train[i][4])
        if distance > speed+2*tolerance_high:
            #print(f"distance too big {distance}")
            #print(i)
            continue
        if (distance>=speed-tolerance_low and distance <= speed+tolerance_high):
            # print(distance)
            start = raw_dataset_train[i]
            d = np.append(d,[start],axis = 0)
            change = speed + random.uniform(bottom, upper)
            if change > speed_max or change < speed_min:
                if case != 2:
                    temp = bottom
                    bottom = -1 * upper
                    upper = -1 * temp
            else :
                speed =  change
    print(f"Generate {len(d)} datapoints")
    return d


def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Input:
      Two data point coordinate
    return:
      Two data point distance 
    """
    radian = 0.0174532925
    lat1 *= radian
    lat2 *= radian
    lon1 *= radian
    lon2 *= radian
    temp = math.sqrt(math.sin(abs(lat1-lat2)/2) * math.sin(abs(lat1-lat2)/2) + math.cos(lat1)*math.cos(lat2) *math.sin(abs(lon1-lon2)/2)*math.sin(abs(lon1-lon2)/2))
    try:
        temp = math.asin(temp)*2
    except ValueError:
        temp = 0
        distance = 1000
    distance = temp*6371.009*1000
    return distance

