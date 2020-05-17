import tensorflow as tf
from util import *
import numpy as np
import argparse
import os
import random
import math
import time
import json

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
with open("hyperparams.json") as f:
	hparams = json.load(f)
speed_scale = hparams['speed_scale']
rot_scale = hparams['rot_scale']

# data
print('loading data')
test_im = np.load('dataset/data/test_im.npy')
test_raw = np.load('dataset/data/test_raw.npy')
# PERFORM PERMUTATION HERE
print('done loading data')
n_data = test_im.shape[0]

# graph
im_in = tf.placeholder(dtype=tf.uint8, shape=[batch_size,64,64,1])
angle_in = tf.placeholder(dtype=tf.float32, shape=[batch_size])

speed_in = tf.placeholder(dtype=tf.float32, shape=[batch_size])
rotation_in = tf.placeholder(dtype=tf.float32, shape=[batch_size])

im = (tf.to_float(im_in)/255.0) - 0.5
ang = tf.expand_dims(angle_in,axis=-1) 
speed = tf.expand_dims(speed_in,axis=-1) * speed_scale
rot = tf.expand_dims(rotation_in,axis=-1) * rot_scale

ang1 = tf.sin(ang*3.14159/180.0)
ang2 = tf.cos(ang*3.14159/180.0)
angNorm = tf.concat([ang1,ang2],axis=-1)

networkBody = NetworkBody(im,angNorm,training=False)

loss = Loss(networkBody.pred, speed,rot)

# saver
saver = tf.train.Saver(max_to_keep=0)

# train
with sessionSetup(cmdArgs) as sess:
	recentSave = cmdArgs.iterations
	saver.restore(sess,'snapshots/iter_'+str(recentSave).zfill(16)+'.ckpt')

	# train loop
	speed_diff = 0.0
	rot_diff = 0.0
	for it in range(n_data):
		# generate batch
		batch_im = test_im[it:it+1]
		batch_raw = test_raw[it:it+1]

		batch_angle = batch_raw[:,0]
		batch_speed = batch_raw[:,1]
		batch_rotation = batch_raw[:,2]

		feed_dict = {
			im_in: batch_im,
			angle_in: batch_angle,
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

	print("Speed avg (cm/s): {}".format(speed_avg*pixel_cm_ratio*fps/speed_scale))
	print("Rot avg (degrees): {}".format(rot_avg/rot_scale))

