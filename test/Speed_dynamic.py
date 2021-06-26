import csv
import numpy
import pandas as pd
import os
import random
import numpy as np
import math
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--step", type=int, default=60)
parser.add_argument("--sample", type=int, default=50, help = 'Sampling rate')
args = parser.parse_args()


def add_gausian(dataset):
    """
    Return:
      Noise + Dataset
    """
    gaussian = np.random.uniform(5,10,dataset.shape)
    dataset_noise  = gaussian + dataset
    return dataset_noise



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

def processing(file_,filepath,filename,add_la,add_lo,scale,step,sample):
    """
    Input:
      param file_ : Now file
      param filapath : Now file path 
      param filename : Now file name (including path)
      param add_la , add_lo , scale : Data preprocessing settings
      param step : How long do we need
      param sample : Sampling rate
    Goal :
      Generate first differential and second differential feature (xyz)
    """
    print(f"Start processing {file_}")
    #---------------Loading data-------------------------------------
    df = pd.read_csv(file_)
    df = df[["MagX", "MagY", "MagZ", "Latitude", "Longitude"]]
    raw_dataset_train = df.to_numpy()
    raw_dataset_train = np.array(raw_dataset_train, np.float64)
    sample_starts = [0, 10, 20, 30]
    for sample_start in sample_starts:
        sample_data = raw_dataset_train[sample_start::sample, :]
        dataset_train = np.array([sample_data[0]])

        for i in range(len(sample_data)-1):
            d = calculate_distance(sample_data[i][3], sample_data[i][4], sample_data[i+1][3], sample_data[i+1][4])
            if d > 0.3:
                dataset_train = np.append(dataset_train,[sample_data[i+1]],axis=0)

        #-----------------------------------Add Noise---------------------------------
        # feature = dataset_train[:,0:len(dataset_train[0])-2]
        # label = dataset_train[:,len(dataset_train[0])-2:len(dataset_train[0])]
        # feature = add_gausian(feature)
        # dataset_train = np.concatenate([feature,label],axis=1)

        # feature = second_dataset_train[:,0:len(second_dataset_train[0])-2]
        # label = second_dataset_train[:,len(second_dataset_train[0])-2:len(second_dataset_train[0])]
        # feature = add_gausian(feature)
        # second_dataset_train = np.concatenate([feature,label],axis=1)

        #================================-Want to evaluate second differential================================
        second_dataset_train = np.zeros((dataset_train.shape[0],dataset_train.shape[1]))
        for i in range(len(second_dataset_train)):
            if i==0:
                continue
            else:
                second_dataset_train[i] = dataset_train[i] - dataset_train[i-1]
        #================================Setting some paramaters================================
        count = 0
        f = open(filename,'a')
        w = csv.writer(f,delimiter=',',lineterminator='\r\n')
        total = dataset_train.shape[0]
        li = [i for i in range((step-1),total)]
        #================================Want to calculate feature================================
        data=[]
        data = np.array(data, np.float64)
        for positions in li:
            #-----------------Positive direction --------------------
            for i in range((positions-(step-1)),positions+1):
                data = np.append(data,dataset_train[i][0]-dataset_train[i-1][0])
                data = np.append(data,dataset_train[i][1]-dataset_train[i-1][1])
                data = np.append(data,dataset_train[i][2]-dataset_train[i-1][2])
                data = np.append(data,second_dataset_train[i][0]-second_dataset_train[i-1][0])
                data = np.append(data,second_dataset_train[i][1]-second_dataset_train[i-1][1])
                data = np.append(data,second_dataset_train[i][2]-second_dataset_train[i-1][2])
                if i == positions:
                    data = np.append(data,(dataset_train[i][3]-add_la)*scale)
                    data = np.append(data,(dataset_train[i][4]-add_lo)*scale)
            w.writerow(data)
            data=[]
            data = np.array(data, np.float64)
            
            

def Main(Rawdata_Dir,filepath_Dest,add_la,add_lo,scale,step,sample):
    for file in os.listdir(Rawdata_Dir):
        filepath = f"./{filepath_Dest}/{file}"
        filename = f"{filepath_Dest}/{file}"
        if os.path.isfile(filepath):
            os.remove(filename)          
        processing(f"{Rawdata_Dir}/{file}",filepath,filename,add_la,add_lo,scale,step,sample)

if __name__ == '__main__':   
          
    
    add_la= 25.08
    add_lo = 121.55
    scale = 10000
    step = args.step
    sample = args.sample    
    print(f"Now processing step {step}")

    Rawdata_Dir = "1029_Miramar"
    filepath_Dest = "1029_fixed_speed_sample"
    Main(Rawdata_Dir,filepath_Dest,add_la,add_lo,scale,step,sample)
    print("10/29 Finish")

    Rawdata_Dir = "1030_Miramar"
    filepath_Dest = "1030_fixed_speed_sample"
    Main(Rawdata_Dir,filepath_Dest,add_la,add_lo,scale,step,sample)
    print("10/30 Finish") 


    Rawdata_Dir = "20201208_miramar_test_round3"
    filepath_Dest = "20201208_speed_sample"
    Main(Rawdata_Dir,filepath_Dest,add_la,add_lo,scale,step,sample)
    print("12/08 Finish") 
    
        


