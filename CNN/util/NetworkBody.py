import tensorflow as tf
from op_blocks import*

class NetworkBody:
	def __init__(self,im,angle,training=False):
		self.build_network(im,angle,training)

	def flatten(self,x):
		x_shape = x.get_shape()
		x_h = x_shape[1].value
		x_w = x_shape[2].value
		x_c = x_shape[3].value
		y = tf.reshape(x,shape=[-1,x_h*x_w*x_c])
		return y

	def build_network(self,im,angle,training):
		#fc = fc_layer(angle,8,activation_function='none')
		#fc = tf.expand_dims(fc,axis=[1])
		#fc = tf.expand_dims(fc,axis=[1])
		#conv0 = conv_2d_layer(im,5,8,1,activation_function='none')
		#conv0 = fc + conv0
		#conv0 = tf.nn.relu(conv0)
		conv0 = conv_2d_layer(im,5,8,1)

		conv1 = conv_2d_layer(conv0,5,16,2)
		conv2 = conv_2d_layer(conv1,5,32,2)
		#conv3 = conv_2d_layer(conv2,5,64,2)

		fl = self.flatten(conv2)
		fl = tf.concat([fl,angle],axis=-1)

		fc0 = fc_layer(fl,128, activation_function='relu')
		fc1 = fc_layer(fc0,32, activation_function='relu')
		fc2 = fc_layer(fc1,2, activation_function='none')

		self.pred = fc2
