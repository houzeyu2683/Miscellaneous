import csv
import pandas as pd
import os
import random
import numpy as np
import math
import random

#-----------------------------------------------------------------------------------
def add_gausian(dataset):
    """
    Return:
      Noise + Dataset
    """
    gaussian = np.random.normal(0,10,dataset.shape)
    dataset_noise  = gaussian + dataset
    return dataset_noise
#-----------------------------------------------------------------------------------
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
#-----------------------------------------------------------------------------------
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
            if change > speed_max or change < speed_min: ##  當速度改變過大或過小，需要調整 uniform 的範圍
                if case != 2:  ##  如果 case 不是 2
                    temp = bottom
                    bottom = -1 * upper
                    upper = -1 * temp
            else :
                speed =  change # 當速度改變的量是可以被接收，那速度就可以更新。
    print(f"Generate {len(d)} datapoints")
    return d
    
def processing(file_,filepath,filename,add_la,add_lo,scale,step,choose_start,target_distance_array,tolerance_low,tolerance_high, speed_max, speed_min, interval):
    """
    Input:
      param file_ : Now file
      param filapath : filename path 
      param filename : Want to Write data here 
      param add_la , add_lo , scale : Data preprocessing settings
      param step : How long do we need
      param choose_start : The start point
      param target_distance_array : The array of target_distance ( def fix_distance will explain)
      param tolerance : Because we can not find the perfect target_distance , so it need some tolerance 
      param speed_max : Upperbound of the speed , avoid driving too fast
      param speed_min : Lowerbound of the speed , avoid driving too slow
      param interval : The allowed speed change in each time

    Goal :
      Generate first differential and second differential feature (xyz) and save it to .csv
      In every path , it will simulate dynamic speed , and every path will have multilples possible  
    """
    print(f"Start processing {file_}")
    #---------------Loading data-------------------------------------
    df = pd.read_csv(file_)
    df = df[["MagX", "MagY", "MagZ", "Latitude", "Longitude"]]
    raw_dataset_train = df.to_numpy()
    raw_dataset_train = np.array(raw_dataset_train, np.float64)
    sample_sequence(raw_dataset_train, filename, add_la,add_lo,scale,step,choose_start,target_distance_array,tolerance_low, tolerance_high,speed_max, speed_min, interval)

def sample_sequence(raw_dataset_train, filename, add_la,add_lo,scale,step,choose_start,target_distance_array,tolerance_low,tolerance_high, speed_max, speed_min, interval):
    """
    same as def processing
    """
    special_value = -1000 # for Masking layer
    for target_distance in target_distance_array:
        print(f"=======================\nNow processing distance {target_distance} \n=======================")
        for s in choose_start: # [0,0,0,0,0]
            for case in range(3): # [0,1,2]
                dataset_train = fixed_distance(raw_dataset_train[s:, :],target_distance, tolerance_low, tolerance_high, speed_max, speed_min, interval, case)
                
                print(dataset_train.shape)
                ##  不懂上面就想成挑了一些點，那些點組成了一個軌跡路徑。

                #================================Want to evaluate second differential================================
                # 這邊看起來像是一次微分，但上述居然寫二次微分，莫非前面找點的方式就是一次微分？
                second_dataset_train = np.zeros((dataset_train.shape[0],dataset_train.shape[1]))
                for i in range(len(second_dataset_train)):
                    if i==0:
                        continue  ##  第一筆資料沒有上一個狀態的座標，所以變化量就是 0 。
                    else:
                        second_dataset_train[i] = dataset_train[i] - dataset_train[i-1]
                
                #---------------Setting some paramaters-------------------------------------
                count = 0
                f = open(filename,'a')
                w = csv.writer(f,delimiter=',',lineterminator='\r\n')

                total = dataset_train.shape[0]
                sample_number= total-step
                if sample_number < 0: ##  太小直接考慮下一個 case
                    continue
                li = [i for i in range((step-1),total)]
                li_length = len(li)
                res = random.sample(li,sample_number)  ##  打散 index
                #---------------Want to calculate feature-------------------------------------
                #---------------data for full_step (60) ----------------------------------------
                #---------------data1 for half_step (30) ----------------------------------------
                data=[]
                data = np.array(data, np.float64)
                data1=[]
                data1 = np.array(data, np.float64)
                for positions in res:  ##  固定一個位置
                    #-----------------Positive direction--------------------
                    #positions is the final data points in this path
                    loop = range((positions-(step-1)),positions+1)  ##  往回推 60 步
                    for cnt, i in enumerate(loop):  ## 對於每一步執行下面
                        if cnt >= step/2: ##  後30個步會新增一個 data1 來存資料，但data持續執行
                            data1 = np.append(data1,dataset_train[i][0]-dataset_train[i-1][0]) # 變化量
                            data1 = np.append(data1,dataset_train[i][1]-dataset_train[i-1][1]) # 變化量
                            data1 = np.append(data1,dataset_train[i][2]-dataset_train[i-1][2]) # 變化量
                            data1 = np.append(data1,second_dataset_train[i][0]-second_dataset_train[i-1][0])
                            data1 = np.append(data1,second_dataset_train[i][1]-second_dataset_train[i-1][1])
                            data1 = np.append(data1,second_dataset_train[i][2]-second_dataset_train[i-1][2])
                            data1_shape = data1.shape
                            print(data1_shape)

                        ##  一定會執行    
                        data = np.append(data,dataset_train[i][0]-dataset_train[i-1][0]) #
                        data = np.append(data,dataset_train[i][1]-dataset_train[i-1][1]) #
                        data = np.append(data,dataset_train[i][2]-dataset_train[i-1][2]) #
                        data = np.append(data,second_dataset_train[i][0]-second_dataset_train[i-1][0])
                        data = np.append(data,second_dataset_train[i][1]-second_dataset_train[i-1][1])
                        data = np.append(data,second_dataset_train[i][2]-second_dataset_train[i-1][2])
                        data_shape = data.shape
                        print(data_shape)

                        ##  這邊看起來要處理 target
                        if i == positions: ##  最後一圈執行
                            data1 = np.append(data1,np.full((int(step/2)*6), fill_value=special_value))         
                            data1 = np.append(data1,(dataset_train[i][3]-add_la)*scale)
                            data1 = np.append(data1,(dataset_train[i][4]-add_lo)*scale)
                            data = np.append(data,(dataset_train[i][3]-add_la)*scale)
                            data = np.append(data,(dataset_train[i][4]-add_lo)*scale)
                    w.writerow(data)  ##  1 row 1 row 的寫進去資料
                    w.writerow(data1)  ##  1 row 1 row 的寫進去資料
                    
                    print("shape of data : {}".format(data.shape))
                    print("shape of data1 : {}".format(data1.shape))
                    print('try understand data and data1.')

                    data=[]
                    data = np.array(data, np.float64)
                    data1=[]
                    data1 = np.array(data, np.float64)


def Suture(files,filename,add_la,add_lo,scale,step,choose_start,target_distance,speed_max, speed_min, interval):
    """
    Input :
      param files : 
      param filename :
      param add_la , add_lo , scale , step : 
      param choose_start : 
      param target_distance : 
      param speed_max : 
      param speed_min : 
      param interval : 
      All param are Same as above
    Settings :
      param Iterval : every Iterval datapoints to see whether it could Suture
      param DistanceThreshold : Condition to find nearest point
      param Angle : Condition to avoid unreasonable path ( usually set 90 )
      param SeeHowManySecond1 : Look back SeeHowManySecond1 seconds
      param SeeHowManySecond2 : Look forward SeeHowManySecond2 seconds
      param tolerance : Because we can not find the perfect target_distance , so it need some tolerance 
    Output :
      Suture many possible paths and generate features into filename
    """
    Iterval = 250
    DistanceThreshold = 3
    Angle = 90
    SeeHowManySecond1 = 6000
    SeeHowManySecond2 = 6000
    start_from = 0
    tolerance_low = 0.5
    tolerance_high = 2
    for f in files :
        start_from = 0
        df = pd.read_csv(f)
        for i in range(int(len(df)/Iterval)):
            if start_from > df.shape[0]-SeeHowManySecond1:
                break
            choose_array = Find_Index(f,start_from,Angle,DistanceThreshold,SeeHowManySecond1)
            for choose in choose_array:
                Dataset = CreateDataset(f,choose,start_from,SeeHowManySecond1,SeeHowManySecond2)
                sample_sequence(Dataset, filename, add_la,add_lo,scale,step, choose_start, target_distance, tolerance_low, tolerance_high, speed_max, speed_min, interval)  
            start_from = start_from+Iterval

def calculate_angle(a,b):
    """
    Input :
      param a : vector
      param b : vector
    Return :
      param angle : angel between two angle
    """
    b= b.T
    ab = a.dot(b)
    len_a = np.sqrt(a[0]**2+a[1]**2)
    len_b = np.sqrt(b[0]**2+b[1]**2)
    cos = ab/(len_a*len_b)
    angle = np.arccos(cos)
    angle = angle / 0.0174532925
    return angle    

def CreateDataset(file,choose,choose_start,SeeHowManySecond1, SeeHowManySecond2):
    """
    Input :
      param file : Which Data
      param choose : From choose array 
      param choose_start : Start point
      param SeeHowManySecond1 : Look back SeeHowManySecond1 seconds
      param SeeHowManySecond2 : Look forward SeeHowManySecond2 seconds
    Output :
      param Dataset : Suture is finished 
    """
    df = pd.read_csv(file)
    df = df[["MagX", "MagY", "MagZ", "Latitude", "Longitude"]]
    raw_dataset_train = df.to_numpy()
    raw_dataset_train = np.array(raw_dataset_train, np.float64)
    
    first_half = choose_start-SeeHowManySecond1
    second_half = choose + SeeHowManySecond1
    
    First_Half = raw_dataset_train[first_half:choose_start,:]
    Second_Half = raw_dataset_train[choose:second_half,:]
    if First_Half.shape[0] < SeeHowManySecond1/2:
        second_half = choose + SeeHowManySecond2
        Second_Half = raw_dataset_train[choose:second_half,:]
    if Second_Half.shape[0] < SeeHowManySecond1/2:
        first_half = choose_start-SeeHowManySecond2
        First_Half = raw_dataset_train[first_half:choose_start,:]
    Dataset = np.concatenate((First_Half,Second_Half),axis = 0)
    return Dataset

def Find_Index(file,choose_start,Angle,DistanceThreshold,SeeHowManySecond):
    """
    Input :
      param file : Which Data
      param choose_start : Start point
      param Angle : Condition to avoid unreasonable path ( usually set 90 )
      param DistanceThreshold : Condition to find nearest point
      param SeeHowManyScond : Decide to how long is the suture
    Output :
      param array : Some index that satisfy all the condition ( Distance , angle ) at the choose_start 
    """

    df = pd.read_csv(file)
    df = df[["MagX", "MagY", "MagZ", "Latitude", "Longitude","Heading"]]
    raw_dataset_train = df.to_numpy()
    raw_dataset_train = np.array(raw_dataset_train, np.float64)
    array = []
    index = 0
    for i in range((raw_dataset_train.shape[0])):
        if (index>raw_dataset_train.shape[0]-101):
            break
        vector1 = raw_dataset_train[choose_start,3:5] - raw_dataset_train[choose_start-1,3:5]
        vector2 = raw_dataset_train[index+100,3:5] - raw_dataset_train[choose_start,3:5]  
        angle = calculate_angle(vector1,vector2)
        if (angle < Angle) :
            angle_condition = True
        else:
            angle_condition = False
            index = index +1
            continue
        condition_distance = calculate_distance(raw_dataset_train[index,3],raw_dataset_train[index,4],raw_dataset_train[choose_start,3],raw_dataset_train[choose_start,4])<DistanceThreshold
        condition_index = abs((index - choose_start)) >SeeHowManySecond
        if (angle_condition ==True & condition_distance==True & condition_index ==True):
            array.append(index)
            index = index+SeeHowManySecond
            continue
        index = index +1
    print(f"Now index is {choose_start}")
    print(f"Find Nearest data point index is {array}")
    return array


if __name__ == '__main__':
    """
    param filepath : the path of filename
    param filename : Want to write feature in it 
    param add_la , add_lo , scale : Data preprocessing settings
    param step : Data preprocessing settings
    param choose_start : The start point
    param target_distance : The array of target_distance ( def fix_distance will explain)
    param tolerance : Because we can not find the perfect target_distance , so it need some tolerance 
    param speed_min : Upperbound of the speed , avoid driving too fast
    param speed_max : Lowerbound of the speed , avoid driving too slow
    param interval :  The allowed speed change in each time
    """                     
    filepath = "./Suture0311.csv"
    filename = "Suture0311.csv"
    if os.path.isfile(filepath):
        os.remove(filename)
    # random.seed(1234)
    random.seed(123456)
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
    for file in os.listdir(Dataset_dir):
        file = os.listdir(Dataset_dir)[0]
        processing(f"{Dataset_dir}/{file}",filepath,filename,add_la,add_lo,scale,step,choose_start,target_distance,tolerance,tolerance, speed_max, speed_min, interval)
    #======================================Want to suture path =================================================
    files = [
        "./Trainingdata/20201029_Miramar_Final_MagMap_NCTU_new.csv",
        "./Trainingdata/20201030_Miramar_Final_MagMap_NCTU_new.csv",
        "./Trainingdata/Final_MagMap_20201029_MiramarB2_B_ANA1.csv",
        "./Trainingdata/Final_MagMap_20201030_MiramarB2_R2_2_J_M20.csv"
        ]
    Suture(files,filename,add_la,add_lo,scale,step,choose_start,target_distance,speed_max, speed_min, interval)