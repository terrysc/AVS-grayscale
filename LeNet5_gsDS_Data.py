#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import time,os,sys
from numba import jit
import matplotlib.pyplot as plt
import pandas as pd
#import mlcompute
#@jit
#def wor():
#	return 0
#

#from tensorflow.python.compiler.mlcompute import mlcompute
#mlcompute.set_mlc_device(device_name='GPU') # Available options are 'cpu', 'gpu', and 'any'.
#physical_devices = tf.config.list_physical_devices()
#print(physical_devices)
#GPUs = tf.config.list_physical_devices('GPU')  # tf2.1版本该函数不再是experimental
#print(GPUs)  # 前面限定了只使用GPU1(索引是从0开始的,本机有2张RTX2080显卡)
#print("Num GPUs Available: ", len(GPUs))
#tf.config.experimental.set_memory_growth(GPUs[0], True)  # 其实gpus本身就只有一个元素  # from keras.datasets import mnist

#with tf.device('/GPU'): 

np.set_printoptions(precision = None)
np.set_printoptions(linewidth = 120)
np.set_printoptions(threshold = np.inf)

flag_method = 1
optimizer_name = 'adam'
optimizer_name = 'rmsprop'
if flag_method == 0:
	Alg_name = 'CNN_simple' + '_' + optimizer_name
elif flag_method == 1:
	Alg_name = 'LeNet5' + '_' + optimizer_name
elif flag_method == 2:
	Alg_name = 'DeepCNN' + '_' + optimizer_name
    

object_scale_list = [1,2,4,8,16,32,64,128,256,512]
runtime = 30
flag_data_change = 1
learning_data_name = 'gsDS_Data_noisetype0'
learning_noise_proportion = 0
if flag_data_change == 0:
    two_data_name_list = [learning_data_name]
    testing_data_name_list = two_data_name_list
    object_scale = object_scale_list[4]
    print(object_scale_list)
    print(object_scale)
elif flag_data_change == 1:
    #testing_data_name_list = ['gsDS_Data_noisetype1', 'gsDS_Data_noisetype2', 'gsDS_Data_noisetype3', 'gsDS_Data_noisetype4']
    testing_data_name_list = ['gsDS_Data_noisetype0']
if testing_data_name_list == ['gsDS_Data_noisetype0']:
    acc_resultChart = np.zeros((len(testing_data_name_list), 1, runtime))
    time_cost = np.zeros((len(testing_data_name_list), 1, runtime))
    testing_noise_proportion_list = [0]
else:
    acc_resultChart = np.zeros((len(testing_data_name_list), 3, runtime))
    time_cost = np.zeros((len(testing_data_name_list), 3, runtime))
    testing_noise_proportion_list = [0.1, 0.2, 0.3]
imagesize = 32
imagescale = imagesize * imagesize
epochs_num = 100
batch_size_num = 100
row = int(runtime + 2)
columns = 2
result_excel = np.zeros((len(testing_data_name_list), len(testing_noise_proportion_list), row, columns))


if flag_data_change == 0:
    datascale = 10000
    whole_datascale = datascale
    learning_datascale = int(whole_datascale * 8 / 10)
    x_src = np.load(learning_data_name + '_' + str(learning_noise_proportion) + '_' + str(datascale) + '_' + str(imagescale) + '_' + str(object_scale) + '_x' + '.npy')
    t_src = np.load(learning_data_name + '_' + str(learning_noise_proportion) + '_' + str(datascale) + '_' + str(imagescale) + '_' + str(object_scale) + '_t' + '.npy')

    x_src = x_src / 255

    train_test_index = np.arange(datascale, dtype=int)
    np.random.shuffle(train_test_index)

    x = x_src[train_test_index[:learning_datascale]]
    t = t_src[train_test_index[:learning_datascale]]
    print(np.shape(x))

elif flag_data_change == 1:
    ### take data of different object size of same type of noise into one array
    learning_datascale = 800
    testing_datascale = 200
    all_datascale = learning_datascale + testing_datascale
    all_learning_datascale = len(object_scale_list) * learning_datascale
    all_testing_datascale = len(object_scale_list) * testing_datascale
    whole_datascale = len(object_scale_list) * all_datascale
    x_learning_alldata = np.zeros((all_learning_datascale, 2, imagesize, imagesize))
    t_learning_alldata = np.zeros((all_learning_datascale, 8))
    for i in range(len(object_scale_list)):
        x_learning_alldata[i * learning_datascale: (i + 1) * learning_datascale, :, :, :] = np.load(learning_data_name + '_' + str(learning_noise_proportion) + '_' + str(learning_datascale) + '_' + str(imagescale) + '_' + str(object_scale_list[i]) + '_x' + '.npy')
        t_learning_alldata[i * learning_datascale: (i + 1) * learning_datascale, :] = np.load(learning_data_name + '_' + str(learning_noise_proportion) + '_' + str(learning_datascale) + '_' + str(imagescale) + '_' + str(object_scale_list[i]) + '_t' + '.npy')
    x = x_learning_alldata
    t = t_learning_alldata
    x = x / 255

x = np.transpose(x,(0,2,3,1))
t = np.argmax(t, axis = 1)

result_index = []
Num_dataset = 0
for i in range(runtime):
    result_index = result_index + [str(i + 1)]
    x_train = x
    t_train = t

    if flag_method == 0:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(8, (3, 3), activation = 'relu', padding = 'same', input_shape = (32, 32, 2)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            # tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
            # tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation = 'relu'),
            # tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(8, activation = 'softmax')
            ])
    elif flag_method == 1:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (5, 5), activation = 'relu', padding = 'same', input_shape = (32, 32, 2)),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(32, (5, 5), activation = 'relu', padding = 'same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation = 'relu'),
            tf.keras.layers.Dense(64, activation = 'relu'),
            tf.keras.layers.Dense(8, activation = 'softmax')
            ])
    elif flag_method == 2:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu', padding = 'same', input_shape = (32, 32, 2)),
            #tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu', padding = 'same'),
            tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu', padding = 'same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation = 'relu'),
            tf.keras.layers.Dense(128, activation = 'relu'),
            tf.keras.layers.Dense(64, activation = 'relu'),
            tf.keras.layers.Dense(8, activation = 'softmax')
            ])
    #model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.compile(optimizer = optimizer_name, loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    #model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print(Alg_name + ' starts learning tasks!!')
    start = time.time()
    #model.fit(x_train, t_train, epochs=50, batch_size=100, verbose=1, validation_data=(x_test,t_test))
    model.fit(x_train, t_train, epochs = epochs_num, batch_size = batch_size_num, verbose = 1)
    for testing_data_name_list_num in range(len(testing_data_name_list)):
        testing_data_name = testing_data_name_list[testing_data_name_list_num]
        for testing_noise_proportion_list_num in range(len(testing_noise_proportion_list)):
            testing_noise_proportion = testing_noise_proportion_list[testing_noise_proportion_list_num]
            if flag_data_change == 0:
                x1 = x_src[train_test_index[learning_datascale:]]
                t1 = t_src[train_test_index[learning_datascale:]]
            elif flag_data_change == 1:
                for object_scale_list_num in range(len(object_scale_list)):
                    x_testing_alldata = np.zeros((all_testing_datascale, 2, imagesize, imagesize))
                    t_testing_alldata = np.zeros((all_testing_datascale, 8))
                    x_testing_alldata[object_scale_list_num * testing_datascale:(object_scale_list_num + 1) * testing_datascale, :, :, :] = np.load(testing_data_name + '_' + str(testing_noise_proportion) + '_' + str(testing_datascale) + '_' + str(imagescale) + '_' + str(object_scale_list[object_scale_list_num]) + '_x' + '.npy')
                    t_testing_alldata[object_scale_list_num * testing_datascale:(object_scale_list_num + 1) * testing_datascale, :] = np.load(testing_data_name + '_' + str(testing_noise_proportion) + '_' + str(testing_datascale) + '_' + str(imagescale) + '_' + str(object_scale_list[object_scale_list_num]) + '_t' + '.npy')
                    x1 = x_testing_alldata
                    t1 = t_testing_alldata
                    x1 = x1 / 255
            x1 = np.transpose(x1,(0,2,3,1))
            t1 = np.argmax(t1, axis = 1)
            x_train = x
            t_train = t
            x_test = x1
            t_test = t1
            print(x_train.shape, t_train.shape, x_test.shape, t_test.shape)
            print(np.shape(acc_resultChart))
            acc_resultChart[testing_data_name_list_num, testing_noise_proportion_list_num, i] = model.evaluate(x_test, t_test, verbose = 0)[1]
            # history = model.fit(x_train, t_train, epochs=50, batch_size = 100, verbose = 1, validation_data=(x_test,t_test))
            end = time.time()
            time_cost[testing_data_name_list_num, testing_noise_proportion_list_num, i] = end - start
            accuracy = acc_resultChart[testing_data_name_list_num, testing_noise_proportion_list_num, i]
            mean_acc_till_now = np.mean(acc_resultChart[testing_data_name_list_num, testing_noise_proportion_list_num, :i+1])
            max_acc_till_now = max(acc_resultChart[testing_data_name_list_num, testing_noise_proportion_list_num, :i+1])
            print('time cost:', str(time_cost[testing_data_name_list_num, testing_noise_proportion_list_num, i]))
            print('accuracy:', str(accuracy))
            print('accuracy_resultChart:', str(acc_resultChart))
            print('The maen accuracy value till now:', str(mean_acc_till_now))
            print('The best accuracy value till now:', str(max_acc_till_now))
            result_excel[testing_data_name_list_num, testing_noise_proportion_list_num, i, 0] = acc_resultChart[testing_data_name_list_num, testing_noise_proportion_list_num, i]
            result_excel[testing_data_name_list_num, testing_noise_proportion_list_num, i, 1] = time_cost[testing_data_name_list_num, testing_noise_proportion_list_num, i]
            #model.summary()
for testing_data_name_list_num in range(len(testing_data_name_list)):
    #testing_data_name = testing_data_name_list[testing_data_name_list_num]
    for testing_noise_proportion_list_num in range(len(testing_noise_proportion_list)):
        #testing_noise_proportion = testing_noise_proportion_list[noise_proportion_list_num]
        result_excel[testing_data_name_list_num, testing_noise_proportion_list_num, runtime, 0] = np.mean(acc_resultChart[testing_data_name_list_num, testing_noise_proportion_list_num, :])
        print(acc_resultChart[testing_data_name_list_num, testing_noise_proportion_list_num, :])
        print('mean this time', acc_resultChart[testing_data_name_list_num, testing_noise_proportion_list_num, :])
        result_excel[testing_data_name_list_num, testing_noise_proportion_list_num, runtime, 1] = np.mean(time_cost[testing_data_name_list_num, testing_noise_proportion_list_num, :])
        result_excel[testing_data_name_list_num, testing_noise_proportion_list_num, runtime + 1, 0] = max(acc_resultChart[testing_data_name_list_num, testing_noise_proportion_list_num, :])
        result_excel[testing_data_name_list_num, testing_noise_proportion_list_num, runtime + 1, 1] = sum(time_cost[testing_data_name_list_num, testing_noise_proportion_list_num, :])

result_header = []
for testing_data_name_list_num in range(len(testing_data_name_list)):
    for testing_noise_proportion_list_num in range(len(testing_noise_proportion_list)):
        result_header_element = []
        list.append(result_header_element, testing_data_name_list[testing_data_name_list_num])
        list.append(result_header_element, '_')
        list.append(result_header_element, str(testing_noise_proportion_list[testing_noise_proportion_list_num]))
        list.append(result_header, ''.join(result_header_element))
        list.append(result_header, 'time_cost')
print('shape(result_header) = ', np.shape(result_header))
print(result_header)
result_index = result_index + ['Mean']
result_index = result_index + ['Best|Total']
result_index = np.array(result_index)

if testing_data_name_list == ['gsDS_Data_noisetype0']:
    if flag_data_change == 0:
        writer_summ_path = './' + Alg_name + '_' + learning_data_name + '_to_gsDS_Data_noisetype0_0' + '_' + str(whole_datascale) + '_' + str(imagescale) + '_' + str(object_scale) + '_epochs' + str(epochs_num) + '_batch' + str(batch_size_num) + '.xlsx'
    elif flag_data_change == 1:
        writer_summ_path = './' + Alg_name + '_' + learning_data_name + '_to_gsDS_Data_noisetype0_0' + '_' + str(whole_datascale) + '_' + str(imagescale) + '_epochs' + str(epochs_num) + '_batch' + str(batch_size_num) + '.xlsx'
else:
    if flag_data_change == 0:
        writer_summ_path = './' + Alg_name + '_' + learning_data_name + '_to_all' + '_' + str(whole_datascale) + '_' + str(imagescale) + '_' + str(object_scale) + '_epochs' + str(epochs_num) + '_batch' + str(batch_size_num) + '.xlsx'
    elif flag_data_change == 1:
        writer_summ_path = './' + Alg_name + '_' + learning_data_name + '_to_all' + '_' + str(whole_datascale) + '_' + str(imagescale) + '_epochs' + str(epochs_num) + '_batch' + str(batch_size_num) + '.xlsx'
writer_summ = pd.ExcelWriter(writer_summ_path)
#result_excel_summ = np.empty(shape = np.shape(result_excel[0,0]))
#result_excel_summ = np.ones(shape = np.shape(result_excel[0,0]))
#print('result_excel_summ:', result_excel_summ)
#print(np.shape(result_excel))
#print('result_excel:', result_excel)
print(np.shape(result_excel)[0])
print(np.shape(result_excel)[1])
row_result_excel_summ = np.shape(result_excel)[2]
col_result_excel_summ = np.shape(result_excel)[0] * np.shape(result_excel)[1] * np.shape(result_excel)[3]
#result_excel_summ = result_excel.reshape((row_result_excel_summ, col_result_excel_summ), order = '')
result_excel_summ = np.zeros((row_result_excel_summ, col_result_excel_summ))
sum_ij = 0
for i in range(np.shape(result_excel)[0]):
    for j in range(np.shape(result_excel)[1]):
        jj1 = (sum_ij + 1) * 2 - 2
        jj2 = (sum_ij + 1) * 2 - 1
        sum_ij = sum_ij + 1
        result_excel_summ[:, jj1] = result_excel[i, j, :, 0]
        result_excel_summ[:, jj2] = result_excel[i, j, :, 1]
        print(sum_ij)
        print(np.shape(result_excel[i,j]))
        print(np.shape(result_excel_summ))
        #result_excel_summ = np.hstack((result_excel_summ, result_excel[i,j,:,:]))
        print('result_excel_summ = ', result_excel_summ)
print('shape(result_excel_summ) = ', np.shape(result_excel_summ))
excel_data_summ = pd.DataFrame(result_excel_summ, index = result_index)
excel_data_summ.to_excel(writer_summ, sheet_name = str(1), index = True, header = result_header)
writer_summ.save()


for testing_data_name_list_num in range(len(testing_data_name_list)):
    print(testing_data_name_list)
    print(testing_data_name_list_num)
    testing_data_name = testing_data_name_list[testing_data_name_list_num]
    for noise_proportion_list_num in range(len(testing_noise_proportion_list)):
        testing_noise_proportion = testing_noise_proportion_list[noise_proportion_list_num]
        if flag_data_change == 0:
            writer_path = './' + Alg_name + '_' + learning_data_name + '_to_' + testing_data_name + '_' + str(testing_noise_proportion) + '_' + str(whole_datascale) + '_' + str(imagescale) + '_' + str(object_scale) + '_epochs' + str(epochs_num) + '_batch' + str(batch_size_num) + '.xlsx'
        elif flag_data_change == 1:
            writer_path = './' + Alg_name + '_' + learning_data_name + '_to_' + testing_data_name + '_' + str(testing_noise_proportion) + '_' + str(whole_datascale) + '_' + str(imagescale) + '_epochs' + str(epochs_num) + '_batch' + str(batch_size_num) + '.xlsx'
        writer = pd.ExcelWriter(writer_path)
        #writer_summ = pd.ExcelWriter(writer_summ_path)
        excel_data = pd.DataFrame(result_excel[testing_data_name_list_num, testing_noise_proportion_list_num, :, :], index = result_index)
        excel_data.to_excel(writer, sheet_name = str(1), index = True, header = ['accuracy' , 'cost_time'])
        #excel_data_summ.to_excel(writer_summ, sheet_name = str(1), index = True, header = result_header)
        writer.save()
        if flag_data_change == 0:
            print(Alg_name + '_' + learning_data_name + '_to_' + testing_data_name + '_' + str(testing_noise_proportion) + '_' + str(whole_datascale) + '_' + str(imagescale) + '_' + str(object_scale) + '_epochs' + str(epochs_num) + '_batch' + str(batch_size_num) + ' test has been finished!!')
        elif flag_data_change == 1:
            print(Alg_name + '_' + learning_data_name + '_to_' + testing_data_name + '_' + str(testing_noise_proportion) + '_' + str(whole_datascale) + '_' + str(imagescale) + '_epochs' + str(epochs_num) + '_batch' + str(batch_size_num) + ' test has been finished!!')






