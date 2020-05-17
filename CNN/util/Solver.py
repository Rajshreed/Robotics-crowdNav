import tensorflow as tf

class Solver:
	def __init__(self, loss):
		self.lr = tf.placeholder(dtype=tf.float32,shape=[])
		solver = tf.train.GradientDescentOptimizer(self.lr)
		#solver = tf.train.MomentumOptimizer(self.lr,0.95)

		self.minimize = solver.minimize(loss)
