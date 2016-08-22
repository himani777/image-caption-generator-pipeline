#! /usr/bin/env python


## pre-process images

import os
import cv2
import numpy as np

path_to_dir = 'raw_data/data3/'


def get_images_and_texts():
	''' reads images from the directory'''
	fils = os.listdir(path_to_dir)

	imgList = []
	images = []

	for fil in fils:
		filn , ext = os.path.splitext(fil)
		if fil.endswith('.jpg'):
			img = cv2.imread(path_to_dir+fil)
			img.resize((3,224,224))
			images.append(img)
			imgList.append(path_to_dir+fil)

	images = np.asarray(images)


	return images


