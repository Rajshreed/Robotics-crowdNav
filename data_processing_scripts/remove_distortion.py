import tensorflow as tf
from PIL import Image
import os

root = '5'
ims = os.listdir('raw/'+root)
ims.sort()

# ------------------------

# load image
fname = tf.placeholder(dtype=tf.string)
im = tf.image.decode_image(tf.read_file(fname))
im = tf.expand_dims(im,0)
im = tf.to_float(im)

# intrinsic
intrinsic = [
	[418.1566,0.0,0.0],
	[-0.2329,417.7446,0.0],
	[970.8695,552.5609,1.0]
]
#intrinsic = tf.linalg.transpose(intrinsic)
intrinsic_i = tf.linalg.inv(intrinsic)

# create meshgrid
cx = 960.0
cy = 540.0
x = list(range(1920))
y = list(range(1080))
x, y = tf.meshgrid(x,y)
grid = tf.stack([x,y],axis=-1)
grid = tf.reshape(grid,[-1,2])
grid = tf.to_float(grid)

# view shift, to viewspace
grid -= [[cx,cy]]
grid *= 0.5
grid += [[cx,cy]]

ones = tf.reshape(tf.ones(shape=tf.shape(x)),[-1,1])
grid = tf.concat([grid,ones],axis=-1)
grid = tf.matmul(grid,intrinsic_i)
grid = grid[:,0:2]

#distortion
r2 = tf.reduce_sum(grid**2,axis=1)

# tangential
p0 = -0.00021189
p1 = -0.000015285
b = 2*p0*tf.reduce_prod(grid,axis=1)
x2 = grid[:,0]**2
b += p1*(r2 + 2.0*x2)
b = tf.expand_dims(b,axis=-1)
grid -= b

# radial
k0 = 0.1022
k1 = 0.00079
k2 = -0.00023525
k = (1.0 + k0*r2 + k1*(r2**2) + k2*(r2**3))
k = 1.0/tf.expand_dims(k,axis=-1)
grid *= k

# to pixel space
grid = tf.concat([grid,ones],axis=-1)
grid = tf.matmul(grid,intrinsic)
grid = grid[:,0:2]

# resample
grid = tf.reshape(grid,[1080,1920,2])
grid = tf.expand_dims(grid,axis=0)
imout = tf.contrib.resampler.resampler(im,grid)
imout = tf.squeeze(imout)
imout = tf.cast(imout,tf.uint8)

with tf.Session() as sess:
	for imname in ims:
		print(imname)
		feed_dict = {
			fname: 'raw/'+root+'/'+imname
		}
		r = sess.run([imout],feed_dict=feed_dict)
		im = Image.fromarray(r[0])
		im.save('dis/'+root+'/'+imname)
