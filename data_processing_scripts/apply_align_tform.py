import tensorflow as tf
from PIL import Image
import random
import numpy as np
import os

scale = 2000

root = '5'
ims = os.listdir('dis/'+root)
ims.sort()

# ------------------------

essential = np.load('tform.npy')
essential = tf.constant(essential)

# load image
fname = tf.placeholder(dtype=tf.string)
im = tf.image.decode_image(tf.read_file(fname))
im = tf.expand_dims(im,0)
im = tf.to_float(im)

# create meshgrid
cx = 960.0
cy = 540.0
x = list(range(1920))
y = list(range(1080))
x, y = tf.meshgrid(x,y)
grid = tf.stack([x,y],axis=-1)
grid = tf.reshape(grid,[-1,2])
grid = tf.to_float(grid)

grid /= scale
ones = tf.reshape(tf.ones(shape=tf.shape(x)),[-1,1])
grid = tf.concat([grid,ones],axis=-1)

grid = tf.matmul(grid,essential)

# resample
z = tf.expand_dims(grid[:,2],axis=-1)
grid = grid[:,0:2]/z
grid *= scale

grid = tf.reshape(grid,[1080,1920,2])
grid = tf.expand_dims(grid,axis=0)
imout = tf.contrib.resampler.resampler(im,grid)
imout = tf.squeeze(imout)
imout = tf.cast(imout,tf.uint8)

with tf.Session() as sess:
	for imname in ims:
		print(imname)
		feed_dict = {
			fname: 'dis/'+root+'/'+imname
		}
		r = sess.run([imout],feed_dict=feed_dict)
		im = Image.fromarray(r[0][0:680,0:1380,:])
		im.save('align/'+root+'/'+imname)


