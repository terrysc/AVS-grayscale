#!/usr/bin/env python3
import numba
import numpy as np
import os
import random

Alg_name = 'gsDS_Data'
datascale = 10000
dire_datascale = int(datascale / 8)
print(type(dire_datascale))
@numba.jit()
def data_generate(datascale,x,y,p,t):
	
	noise_proportion = p
	object_scale = x
	image_size = y
	noise_type = t
	
	data = np.zeros((datascale, 2, image_size, image_size))
#	print(data.shape)
	
	label = np.zeros((datascale, 8))
	
	 
	
	for direction in range(8):
		
		if direction == 0:
			fang = 0
			xiang = 1  # 右
		if direction == 1:
			fang = -1
			xiang = 1  # 右上
		if direction == 2:
			fang = -1
			xiang = 0  # 上
		if direction == 3:
			fang = -1
			xiang = -1  # 左上
		if direction == 4:
			fang = 0
			xiang = -1  # 左
		if direction == 5:
			fang = 1
			xiang = -1  # 左下
		if direction == 6:
			fang = 1
			xiang = 0  # 下
		if direction == 7:
			fang = 1
			xiang = 1  # 右下
			
			type(direction)
		for i in range(direction * dire_datascale, dire_datascale * (direction + 1)):
			#print(i)

			# Generating Background Pixels
			background = np.zeros((image_size, image_size))
			background_grayscale = np.random.randint(0, 256)
			for ii in range(np.size(background, 0)):
				for jj in range(np.size(background, 1)):
					background[ii, jj] = background_grayscale
			#background = np.ones((image_size * image_size)) * background_grayscale
				#print(np.shape(background))
			#if noise_type == 0:
				background_1 = background
				background_2 = background
				#if noise_proportion != 0:
			#print(background)
			if noise_type == 1:
				noise_pixel_num = noise_proportion * image_size * image_size
				noise_pixel_num = int(noise_pixel_num)
				index_1d = list(range(image_size * image_size))
				random.shuffle(index_1d)
				for ii in range(noise_pixel_num):
					noise_pixel_row = index_1d[ii] // image_size
					noise_pixel_col = index_1d[ii] % image_size
					background_1[noise_pixel_row, noise_pixel_col] = np.random.randint(0, 256)
					background_2 = background_1
			elif noise_type == 2:
				noise_pixel_num = noise_proportion * image_size * image_size
				noise_pixel_num = int(noise_pixel_num)
				index_1d_1 = list(range(image_size * image_size))
				random.shuffle(index_1d_1)
				for ii in range(noise_pixel_num):
					noise_pixel_row = index_1d_1[ii] // image_size
					noise_pixel_col = index_1d_1[ii] % image_size
					background_1[noise_pixel_row, noise_pixel_col] = np.random.randint(0, 256)
				index_1d_2 = list(range(image_size * image_size))
				random.shuffle(index_1d_2)
				for ii in range(noise_pixel_num):
					noise_pixel_row = index_1d_2[ii] // image_size
					noise_pixel_col = index_1d_2[ii] % image_size
					background_2[noise_pixel_row, noise_pixel_col] = np.random.randint(0, 256)
			data[i, 0] = background_1
			data[i, 1] = background_2
			#print(data[i, 0])
			#print(data[i, 1])

			# Generating Object Pixels	
			flag = 1
			while flag == 1:
				object_grayscale = np.random.randint(0, 256)
				if object_grayscale != background_grayscale:
					flag = 0
			
			label[i, direction] = 1 # 第i张图的第direc个位置置为1，其余7个仍为0
			
			h = np.random.randint(1, image_size - 1)
			w = np.random.randint(1, image_size - 1)
			data[i, 0, h, w] = 255 * 9 ###set the value to a large number, for index 
			data[i, 1, h + fang, w + xiang] = 255 * 9
			
			for j in range(object_scale - 1):
				flag = 1
				while flag:
					h = np.random.randint(1, image_size - 1)
					w = np.random.randint(1, image_size - 1)
					if data[i, 0, h, w] != 255 * 9 and np.sum(data[i, 0, h-1:h+2, w-1:w+2]) >= 255 * 9:
						data[i, 0, h, w] = 255 * 9
						data[i, 1, h + fang, w + xiang] = 255 * 9
						flag = 0
			data[data == 255 * 9] = object_grayscale  #### change 255*9 to the random object pixel value
			if noise_type == 3:
				noise_pixel_num = noise_proportion * image_size * image_size
				noise_pixel_num = int(noise_pixel_num)
				index_1d = list(range(image_size * image_size))
				random.shuffle(index_1d)
				for ii in range(noise_pixel_num):
					noise_pixel_row = index_1d[ii] // image_size
					noise_pixel_col = index_1d[ii] % image_size
					data[i, :, noise_pixel_row, noise_pixel_col] = np.random.randint(0, 256)
			elif noise_type == 4:
				noise_pixel_num = noise_proportion * image_size * image_size
				noise_pixel_num = int(noise_pixel_num)
				index_1d_1 = list(range(image_size * image_size))
				random.shuffle(index_1d_1)
				for ii in range(noise_pixel_num):
					noise_pixel_row = index_1d_1[ii] // image_size
					noise_pixel_col = index_1d_1[ii] % image_size
					data[i, 0, noise_pixel_row, noise_pixel_col] = np.random.randint(0, 256)
				index_1d_2 = list(range(image_size * image_size))
				random.shuffle(index_1d_2)
				for ii in range(noise_pixel_num):
					noise_pixel_row = index_1d_2[ii] // image_size
					noise_pixel_col = index_1d_2[ii] % image_size
					data[i, 1, noise_pixel_row, noise_pixel_col] = np.random.randint(0, 256)

			#print(data[i, 0])
			#print(data[i, 1])
	return data, label

noise_type_list = [0,1,2,3,4]
noise_type_list = [4]
for noise_type in noise_type_list:
	print('noise_type = ', str(noise_type))
	noise_proportion_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	noise_proportion_list = [0.1, 0.2, 0.3]

	if noise_type == 0:
		noise_proportion_list = [0]
	elif noise_type == 1: # Static Background Noise
		noise_proportion_list = [0.1, 0.2, 0.3]
	elif noise_type == 2: # Dynamic Background Noise
		noise_proportion_list = [0.1, 0.2, 0.3]
	elif noise_type == 3: # Static Wholeimage Noise
		noise_proportion_list = [0.1, 0.2, 0.3]
	elif noise_type == 4: # Dynamic Wholeimage Noise
		noise_proportion_list = [0.1, 0.2, 0.3]

	for noise_proportion in noise_proportion_list:
		print('noise_proportion is now ',  str(noise_proportion))
		object_size_list = [1,2,4,8,16,32,64,128,256,512]
		#object_size_list = [1]
		for object_scale in object_size_list:
			print('object_scale is now ',  str(object_scale))
			image_size = 32
			imagescale = image_size * image_size
			#print(type(object_scale))
			#print(type(image_size))
			data, label = data_generate(datascale, object_scale, image_size, noise_proportion, noise_type)
			dir_path = os.getcwd()

			np.save(str(dir_path) + '/' + Alg_name + '_noisetype' + str(noise_type) + '_' + str(noise_proportion) + '_' + str(datascale) + '_' + str(imagescale) + '_' + str(object_scale) + '_x.npy', data)
			np.save(str(dir_path) + '/' + Alg_name + '_noisetype' + str(noise_type) + '_' + str(noise_proportion) + '_' + str(datascale) + '_' + str(imagescale) + '_' + str(object_scale) + '_t.npy', label)
			print(Alg_name + '.py_' + 'object_scale ' + str(object_scale) + ' have been done!')