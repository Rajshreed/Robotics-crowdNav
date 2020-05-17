import tensorflow as tf

class Loss:
	def __init__(self,y,speed,rot):
		with tf.variable_scope(None,default_name='loss'):

			target = tf.concat([speed,rot],axis=-1)
			diff = y - target
			diffsq = diff**2.0

			avgLoss = tf.reduce_mean(diffsq)

			# weight decay
			wd = 0
			for w in tf.get_collection('weight_decay'):
				wd += tf.reduce_sum(tf.abs(w))

			self.loss = avgLoss# + (wd*0.005)

			# accuracy
			self.speed_diff = tf.abs(diff[:,0])
			self.rot_diff = tf.abs(diff[:,1])
