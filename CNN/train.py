import tensorflow as tf
from util import *
import numpy as np
import argparse
import os
import random
import math
import time
import threading
import json

from util import *

# settings
argParser = argparse.ArgumentParser(description="start/resume training")
argParser.add_argument("-l","--logDev",dest="logDev",action="store_true")
argParser.add_argument("-g","--gpu",dest="gpu",action="store",default=1,type=int)
argParser.add_argument("-i","--iterations",dest="iterations",action="store",default=0,type=int)
argParser.add_argument("-r","--resume",dest="resume",action="store_true")
cmdArgs = argParser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(cmdArgs.gpu)

with open("hyperparams.json") as f:
        hparams = json.load(f)
speed_scale = hparams['speed_scale']
rot_scale = hparams['rot_scale']

print_freq = 100
snap_freq = 10000

batch_size = 128
iterations = 100000
lr = 0.0001

# data
print('loading data')
train_im = np.load('dataset/data/train_im.npy')
train_raw = np.load('dataset/data/train_raw.npy')
# PERFORM PERMUTATION HERE
print('done loading data')

n_data = train_raw.shape[0]

# placeholders
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

# network
networkBody = NetworkBody(im,angNorm,training=True)

loss = Loss(networkBody.pred, speed, rot)
solver = Solver(loss.loss)

# saver
saver = tf.train.Saver(max_to_keep=0)

def lr_decay(lr,it):
	return lr*math.exp(-it*0.00005)

# train
with sessionSetup(cmdArgs) as sess:
	sess.run(tf.global_variables_initializer())

	# train loop
	last_it = time.time()
	for it in range(iterations):
		# generate batch
		idxs = random.sample(range(n_data),batch_size)
		batch_im = train_im[idxs]
		batch_raw = train_raw[idxs]

		batch_angle = batch_raw[:,0]
		batch_speed = batch_raw[:,1]
		batch_rotation = batch_raw[:,2]

		adj_lr = lr_decay(lr,it)
		feed_dict = {
			im_in: batch_im,
			angle_in: batch_angle,
			speed_in: batch_speed,
			rotation_in: batch_rotation,
			solver.lr: adj_lr
		}
		r = sess.run([loss.loss, solver.minimize],feed_dict=feed_dict)

		if it % print_freq == 0:
			out_txt = 'It: {} Loss: {}, LR: {}'.format(it,r[0],adj_lr)
			print(out_txt)

		if (it+1) % snap_freq == 0:
			saver.save(sess,"snapshots/iter_"+str(it+1).zfill(16)+".ckpt")







