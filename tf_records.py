from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import logging
import os
import sys

from lxml import etree
import PIL.Image
import tensorflow as tf

sys.path.append('/usr/local/lib/python2.7/dist-packages/tensorflow/models/research/')

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

from scipy.io import loadmat
import cv2
import numpy as np

import json

data = json.load(open('LARA_GT.json'))
data = data['dataset']

train_writer = tf.python_io.TFRecordWriter('LARA_train.record')

def get_example(index, num_obj):
	print(index,num_obj)
	img_id = index
	img_name = 'frame_' + '0' * (6 - len(str(img_id))) + str(img_id) + '.jpg'
	img_path='LARA/'
	img=cv2.imread(img_path + img_name)
	rows, cols, _ = np.shape(img)
	with tf.gfile.GFile(img_path + img_name) as fid:
		encoded_jpg = fid.read()
	encoded_jpg_io = io.BytesIO(encoded_jpg)
	image = PIL.Image.open(encoded_jpg_io)
	key = hashlib.sha256(encoded_jpg).hexdigest()

	width = cols
	height = rows
	tag = bytes(img_id)
	xmin = []
	ymin = []
	xmax = []
	ymax = []
	classes = []
	classes_text = []
	truncated = []
	poses = []
	difficult_obj = []

	if(num_obj > 1):
		for i in range(0, num_obj):
			box = data['frame'][index]['objectlist']['object'][i]['box']
			XMIN = (float(box['_xc']) - (float(box['_w']) / 2)) / width
			XMAX = (float(box['_xc']) + (float(box['_w']) / 2)) / width
			YMIN = (float(box['_yc']) - (float(box['_h']) / 2)) / height
			YMAX = (float(box['_yc']) + (float(box['_h']) / 2)) / height
			if(XMIN >= 1.0 or XMAX >= 1.0 or YMIN >= 1.0 or YMAX >= 1.0 or XMIN < 0 or XMAX < 0 or YMIN < 0 or YMAX < 0):
				print("got")
				return tf.train.Example(),0
			difficult_obj.append(0)
			xmin.append(XMIN)
			xmax.append(XMAX)
			ymin.append(YMIN)
			ymax.append(YMAX)
			classes_text.append("Traffic_light")
			classes.append(1)
			truncated.append(0)
			poses.append('Unspecified')


	else:
		box = data['frame'][index]['objectlist']['object']['box']
		XMIN = (float(box['_xc']) - (float(box['_w']) / 2)) / width
		XMAX = (float(box['_xc']) + (float(box['_w']) / 2)) / width
		YMIN = (float(box['_yc']) - (float(box['_h']) / 2)) / height
		YMAX = (float(box['_yc']) + (float(box['_h']) / 2)) / height
		if(XMIN >= 1.0 or XMAX >= 1.0 or YMIN >= 1.0 or YMAX >= 1.0 or XMIN < 0 or XMAX < 0 or YMIN < 0 or YMAX < 0):
			print("GOT")
			return tf.train.Example(),0
		difficult_obj.append(0)
		xmin.append(XMIN)
		xmax.append(XMAX)
		ymin.append(YMIN)
		ymax.append(YMAX)
		classes_text.append("Traffic_light")
		classes.append(1)
		truncated.append(0)
		poses.append('Unspecified')

	for i in xmin:
		if(i >= 1 or i < 0):
			print("GOT xmin",img_id)
	for i in xmax:
		if(i >= 1 or i < 0):
			print("GOT xmax",img_id)
			print(xmax)
			print(num_obj)
	for i in ymin:
		if(i >= 1 or i < 0):
			print("GOT ymin",img_id)
	for i in ymax:
		if(i >= 1 or i < 0):
			print("GOT ymax",img_id)

	# print(xmin)
	example = tf.train.Example(features=tf.train.Features(feature={
	  'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(tag),
      'image/source_id': dataset_util.bytes_feature(tag),
      'image/key/sha256': dataset_util.bytes_feature(key),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature(bytes('jpg')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(bytes(classes_text)),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(bytes(poses))}))
	return example,1

tl=0
ntl=0
for i in range(0,len(data['frame'])):
	if(i % 10 != 0):
		continue
	frame = data['frame'][i]
	if(frame['objectlist'] == '\n'):
		ntl += 1
		continue
	if(type(frame['objectlist']['object']) == list):
		tf_example,flag = get_example(i,len(frame['objectlist']['object']))
	else:
		tf_example,flag = get_example(i,1)
	if(flag == 1):
		print("Count : ",i)
		train_writer.write(tf_example.SerializeToString())
		tl += 1

print("TL : ",tl," NTL ",ntl)