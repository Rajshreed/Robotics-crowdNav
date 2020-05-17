import tensorflow as tf
from util import *
import numpy as np
import argparse
import os
import random
import math
import time

from util import *

# settings
argParser = argparse.ArgumentParser(description="start/resume training")
argParser.add_argument("-l","--logDev",dest="logDev",action="store_true")
argParser.add_argument("-g","--gpu",dest="gpu",action="store",default=0,type=int)
argParser.add_argument("-i","--iterations",dest="iterations",action="store",default=1,type=int)
argParser.add_argument("-r","--resume",dest="resume",action="store_true")
cmdArgs = argParser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(cmdArgs.gpu)

batch_size = 1

# data
print('loading data')
test_raw = np.load('dataset/data/test_raw.npy')
# PERFORM PERMUTATION HERE
print('done loading data')
n_data = test_raw.shape[0]

# graph
speed_in = tf.placeholder(dtype=tf.float32, shape=[batch_size])
rotation_in = tf.placeholder(dtype=tf.float32, shape=[batch_size])

speed = tf.expand_dims(speed_in,axis=-1) * 1.0
rot = tf.expand_dims(rotation_in,axis=-1) * 1.0

networkBody = lambda: None
networkBody.pred = [2.49460102884,0.886144867864]
networkBody.pred = tf.expand_dims(networkBody.pred,axis=0)

loss = Loss(networkBody.pred, speed,rot)


# train
with sessionSetup(cmdArgs) as sess:
	# train loop
	speed_diff = 0.0
	rot_diff = 0.0
	for it in range(n_data):
		# generate batch
		batch_raw = test_raw[it:it+1]

		batch_speed = batch_raw[:,1]
		batch_rotation = batch_raw[:,2]

		feed_dict = {
			speed_in: batch_speed,
			rotation_in: batch_rotation
		}
		r = sess.run([loss.speed_diff, loss.rot_diff],feed_dict=feed_dict)
		speed_diff += r[0]
		rot_diff += r[1]

		speed_avg = speed_diff/float(it+1)
		rot_avg = rot_diff/float(it+1)
		print("Speed avg: {}, Rotation avg: {}".format(speed_avg,rot_avg))


	print('------------------------------------------------')
	tile_cm = 50.0
	tile_pixels = 100.0
	pixel_cm_ratio = tile_cm/tile_pixels
	fps = 30.0

	print("Speed avg (cm/s): {}".format(speed_avg*pixel_cm_ratio*fps))
	print("Rot avg (degrees): {}".format(rot_avg))

