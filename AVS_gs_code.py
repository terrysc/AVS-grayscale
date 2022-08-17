#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import numba
from PIL import Image

np.set_printoptions(linewidth = 500)  
np.set_printoptions(threshold = np.inf) 


threshold = 0

@numba.jit
def modtion_detection(x1,x2): # x1 is image inputed
    h,w = x1.shape  

    d = np.zeros(8) # d is the detection result (1,0,0,0,0,0,0,0)
    all_cell = np.zeros((8,h,w))

    for i in range(1,h-1):
        for j in range(1,w-1):
            local1 = x1[i-1:i+2,j-1:j+2]
            local2 = x2[i-1:i+2,j-1:j+2]
            refer_pixel = local1[1,1]
           
            if abs(local2[1,1] - refer_pixel) > threshold:
                if abs(local2[1,2] - refer_pixel) <= threshold: all_cell[0,i,j] = 1 # rightwards
                if abs(local2[0,2] - refer_pixel) <= threshold: all_cell[1,i,j] = 1 # upper rightwards
                if abs(local2[0,1] - refer_pixel) <= threshold: all_cell[2,i,j] = 1 # upwards
                if abs(local2[0,0] - refer_pixel) <= threshold: all_cell[3,i,j] = 1 # upper leftwards
                if abs(local2[1,0] - refer_pixel) <= threshold: all_cell[4,i,j] = 1 # leftwards
                if abs(local2[2,0] - refer_pixel) <= threshold: all_cell[5,i,j] = 1 # lower leftwards
                if abs(local2[2,1] - refer_pixel) <= threshold: all_cell[6,i,j] = 1 # downwards
                if abs(local2[2,2] - refer_pixel) <= threshold: all_cell[7,i,j] = 1 # lower rightward

    for i in range(8):
        d[i] = np.sum(all_cell[i])

    return d
    




## main.py ##

Alg_name = 'ARVS_' + str(threshold)
data_name_list = ['gsDS_Data_noisetype0_', 'gsDS_Data_noisetype1_', 'gsDS_Data_noisetype2_', 'gsDS_Data_noisetype3_', 'gsDS_Data_noisetype4_']
data_name_list = ['gsDS_Data_noisetype1_']
for data_name in data_name_list:
    datascale = 10000
    #datascale = 100
    imagescale = 1024
    #noise_proportion_list = [0, 0.1, 0.2, 0.3]
    if data_name == 'gsDS_Data_noisetype0_':
        noise_proportion_list = [0]
    else:
        noise_proportion_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        noise_proportion_list = [0.1, 0.2, 0.3]

    for noise_proportion in noise_proportion_list:
        writer = pd.ExcelWriter('./' + Alg_name + '_' + data_name + str(noise_proportion) + '_' + str(datascale) + '_' + str(imagescale) + '.xlsx')
        object_scale_list = [1,2,4,8,16,32,64,128,256,512]
        #object_scale_list = [32,2,4,8,16,32]
        Num_dataset = 0
        result_excel = np.zeros([len(object_scale_list), 3])
        index = []
        for object_scale in object_scale_list:
            #writer = pd.ExcelWriter('./' + Alg_name + '_' + data_name + str(noise_proportion) + '_' + str(datascale) + '_' + str(imagescale) + '_' + str(object_scale) + '.xlsx')
            #data = np.load('/Volumes/Motion data/dataset/2D motion/32x32 size/p noise 2/x_128pixel_0%contactpnoise.npy')
            #label = np.load('/Volumes/Motion data/dataset/2D motion/32x32 size/p noise 2/t_128pixel_0%contactpnoise.npy')
            index = index + [str(object_scale)]

            Dir_path = os.getcwd()
            data = np.load(str(Dir_path) + '/' + data_name + str(noise_proportion) + '_' + str(datascale) + '_' + str(imagescale) + '_' + str(object_scale) + '_x' + '.npy')
            label = np.load(str(Dir_path) + '/' + data_name + str(noise_proportion) + '_' + str(datascale) + '_' + str(imagescale) + '_' + str(object_scale) + '_t' + '.npy')
            print(data.shape)
            n,c,h,w = data.shape # (10000, 2, 32, 32)
            result = np.zeros((n, 8))
            for i in range(n):
                xx1 = data[i,0]  #  xx1 = data[i,0,:,:]
                xx2 = data[i,1]  #  xx2 = data[i,1,:,:]
                result[i] = modtion_detection(xx1, xx2)
                print(i)
             
            good = 0
            for i in range(n):
                if np.argmax(result[i]) == np.argmax(label[i]): # (100, 35, 24, 65, 78, 90, 21, 3)     #看哪个细胞激活最多
                    good = good + 1
                else:
                    print(i)
                    print(label[i])
                    print(result[i])
                    #  plt.imshow(data[i,0])
                    #  plt.show()
                    #  plt.imshow(data[i,1])
                    #  plt.show()
                    #print(result)
            print(good)
            accuracy = good/n
            print(accuracy)
            result_excel[Num_dataset, 0] = good
            result_excel[Num_dataset, 1] = n
            result_excel[Num_dataset, 2] = accuracy
            Num_dataset = Num_dataset + 1
        index = np.array(index)
        result_excel = pd.DataFrame(result_excel, index = index)
        result_excel.to_excel(writer, sheet_name = str(1), index = True, header=['good' , 'count' , 'accuracy'])
        writer.save()
