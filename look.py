

"""
fixed_distance
"""
import pandas as pd
import random
import numpy as np
import random
df = pd.read_csv("train/Final_MagMap_20201030_MiramarB2_R2_2_J_M20.csv")
df = df[["MagX", "MagY", "MagZ", "Latitude", "Longitude"]]
raw_dataset_train = df.to_numpy()
raw_dataset_train = np.array(raw_dataset_train, np.float64)
tolerance = 0.05
tolerance_low = tolerance
tolerance_high = tolerance
speed_max = 7
speed_min = 0.3
interval = 0.5
case = [0,1,2]
##
##  
d = [raw_dataset_train[0]]
d = np.array(d,np.float64)
start = raw_dataset_train[0]
# max_distance = 0
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