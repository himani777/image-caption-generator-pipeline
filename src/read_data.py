#! /usr/bin/env python


## pre-process images

import os
import cv2
import numpy as np

path_to_dir = 'raw_data/data3/'


def get_images_and_texts():
	''' reads images and texts from the directory'''
	fils = os.listdir(path_to_dir)

	imgList = []
	images = []
	textList = []

	for fil in fils:
		filn , ext = os.path.splitext(fil)
		if ext == '.jpg':
			img = cv2.imread(path_to_dir+fil)
			img.resize((3,224,224))
			images.append(img)
			imgList.append(path_to_dir+filn)
			textFileName = path_to_dir + filn + '.txt'
			f = open(textFileName,'r')
			text = f.read()
			textList.append(text)

	images = np.asarray(images)


	return images, textList


