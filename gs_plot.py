
import matplotlib.pyplot as plt
import matplotlib.axes as axs
import numpy as np
import os
import pandas as pd

def modtion_detection(x1,x2): # x1 is image inputed
    threshold = 0
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
    recorder = np.ones((8, h*w)) * -999999
    #color_int_1 = [int('181'), int('73'), int('91')]
    #color_int_2 = [int('74'), int('89'), int('61')]
    color_int_1 = [0, 0, 0]
    color_int_2 = [0.09, 0.645, 0.935]
    allpixels_activated = np.ones((h, w, 3)) * color_int_1
    print(np.shape(allpixels_activated))
    for i in range(8):
        #allpixels_activated[all_cell[i] == 1] = all_cell[i][all_cell[i] == 1]
        #allpixels_activated[all_cell[i] == 1] = ([int(180),int(129),int(187)])
        allpixels_activated[all_cell[i] == 1] = (color_int_2)
        d[i] = np.sum(all_cell[i])
        LMDN_act_plot_flag = 1
        #recorder = []
        if LMDN_act_plot_flag == 1:
            LMDN_act_1D_data = np.reshape(all_cell[i], -1)
            for ii in range(np.size(LMDN_act_1D_data, 0)):
                if LMDN_act_1D_data[ii] == 1:
                    #recorder = np.append(recorder, [ii])
                    recorder[i][ii] = ii

            #print(recorder)
            #print(type(recorder))
        #print('LMDN number ', str(i), 'has been actvated as a degree of ', str(d[i]))
    #print(d)
    return d, recorder, allpixels_activated

data_name_list = ['gsDS_Data_noisetype0_', 'gsDS_Data_noisetype1_', 'gsDS_Data_noisetype2_', 'gsDS_Data_noisetype3_', 'gsDS_Data_noisetype4_']
data_name_list = ['gsDS_Data_noisetype3_']
for data_name in data_name_list:
    whole_datascale = 10000
    imagescale = 1024
    #noise_proportion_list = [0, 0.1, 0.2, 0.3]
    if data_name == 'gsDS_Data_noisetype1_':
        noise_proportion_list = [0]
    else:
        noise_proportion_list = [0.1, 0.2, 0.3]

    for noise_proportion in noise_proportion_list:
        #writer = pd.ExcelWriter('./' + Alg_name + '_' + data_name + str(noise_proportion) + '_' + str(whole_datascale) + '_' + str(imagescale) + '.xlsx')
        object_scale_list = [1,2,4,8,16,32,64,128,256,512]
        object_scale_list = [16]
        #Num_dataset = 0
        #result_excel = np.zeros([len(object_scale_list), 3])
        index = []
        for object_scale in object_scale_list:
            #index = index + [str(object_scale)]

            Dir_path = os.getcwd()
            data = np.load(str(Dir_path) + '/' + data_name + str(noise_proportion) + '_' + str(whole_datascale) + '_' + str(imagescale) + '_' + str(object_scale) + '_x' + '.npy')
            label = np.load(str(Dir_path) + '/' + data_name + str(noise_proportion) + '_' + str(whole_datascale) + '_' + str(imagescale) + '_' + str(object_scale) + '_t' + '.npy')
            #print(data.shape)
            #print(label.shape)
            #print(type(data))
            #print(type(label))
            #print(label)
            random_ordinal = np.random.randint(0, whole_datascale)
            
            xx1 = data[random_ordinal,0]  #  xx1 = data[i,0,:,:]
            xx2 = data[random_ordinal,1]  #  xx2 = data[i,1,:,:]
            result, recorder, allpixels_activated = modtion_detection(xx1, xx2)

            if np.argmax(result) == np.argmax(label[random_ordinal]):
                index_dire = list(label[random_ordinal]).index(1)
                if index_dire == 0:
            	    Motion_name = 'rightwards'
                elif index_dire == 1:
            	    Motion_name = 'upper rightwards'
                elif index_dire == 2:
            	    Motion_name = 'upwards'
                elif index_dire == 3:
            	    Motion_name = 'upper leftwards'
                elif index_dire == 4:
            	    Motion_name = 'leftwards'
                elif index_dire == 5:
            	    Motion_name = 'lower leftwards'
                elif index_dire == 6:
            	    Motion_name = 'downwards'
                elif index_dire == 7:
            	    Motion_name = 'lower rightward'
                print('Right Discrimination !!! The Motion Direction is ', Motion_name)
                for i in range(8):
                    print('LMDN number ', str(i), 'has been actvated as a degree of ', str(result[i]))
                    #print(recorder[i] == 1)
                    #print(recorder[i].shape)
                    fig, ax = plt.subplots()
                    ax.eventplot(recorder[i], lineoffsets = 0, linewidth = 0.25)
                    #plt.plot([recorder.min()-100, recorder.max()+100], [0, 0])
                    #plt.xlim(recorder.min()-100, recorder.max()+100)
                    plt.plot([-10, 1034], [0, 0])
                    plt.xlim(-10, 1034)
                    plt.show()

                image2plot = data[random_ordinal,0]
                #print(image2plot)
                #print(data[random_ordinal,0])
                #print(data[random_ordinal,1])
                #image2plot_2D = np.reshape(image2plot, -1)
                plt.imshow(image2plot, cmap = 'gray', vmin=0, vmax=255)
                plt.show()
                plt.imshow(data[random_ordinal,1], cmap = 'gray', vmin=0, vmax=255)
                plt.show()
                print(allpixels_activated)
                #plt.imshow(allpixels_activated, cmap = 'gray', vmin=0, vmax=255)
                plt.imshow(allpixels_activated, vmin=0, vmax=255)
                plt.show()
                #plt.style.use('_mpl-gallery')
                fig, ax = plt.subplots()
            else:
            	print('Failed the Discrimination !!! Run again to Get a New Detection.')