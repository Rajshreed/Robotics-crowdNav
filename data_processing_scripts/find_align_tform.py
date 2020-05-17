import tensorflow as tf
from PIL import Image
import random
import numpy as np

n = 80000
lr = 0.000000005
scale = 2000

# load image
fname = tf.placeholder(dtype=tf.string)
im = tf.image.decode_image(tf.read_file(fname))
im = tf.expand_dims(im,0)
im = tf.to_float(im)

# intrinsic
mask = [
	[1.0,1.0,0.0],
	[1.0,1.0,0.0],
	[1.0,1.0,0.0]
]
mask = tf.constant(mask)
essential = [
	[1.0,0.0,0.0],
	[0.0,1.0,0.0],
	[0.0,0.0,1.0]
]
for row in range(3):
	for col in range(3):
		essential[row][col] += random.uniform(-0.1, 0.1)
essential = tf.Variable(essential)

#essential = essential*mask + tf.stop_gradient(essential*(1.0-mask))
# ------------------------------------------
target = [
        [632.0,90.0],
        [1274.0,88.0],
        [446.0,974.0],
        [1490.0,943.0]
]
source=[
	[500.0,400.0],
	[1400.0,400.0],
	[500.0,1500.0],
	[1400.0,1500.0]
]
source = tf.constant(source)
target = tf.constant(target)

source /= scale

sh = tf.concat([source,tf.ones(shape=[4,1])],axis=-1)

shat = tf.matmul(sh,essential)
z = tf.expand_dims(shat[:,2],axis=-1)
shat = shat[:,0:2] / z

shat *= scale

loss = (target - shat)**2
loss = tf.reduce_sum(loss,axis=-1)
loss = tf.sqrt(loss)
loss = tf.reduce_sum(loss)
op = tf.train.MomentumOptimizer(lr,0.9).minimize(loss)
# ------------------------------------------

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
	sess.run(tf.global_variables_initializer())

	for i in range(n):
		r = sess.run([op,loss])
		if i%1000 == 0:
			print(r[1])

	feed_dict = {
		fname: "wow.png"
		#fname: "sample.png"
	}
	r = sess.run([imout],feed_dict=feed_dict)
	im = Image.fromarray(r[0])
	im.save('cool.png')

	r = sess.run(essential)
	np.save('essential3.npy',r)
