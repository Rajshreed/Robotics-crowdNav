import numpy as np


test_raw = np.load('dataset/data/test_raw.npy')

speed = test_raw[:,1]
rot = test_raw[:,2]

print("speed {}".format(np.mean(speed)))
print("rotation {}".format(np.mean(rot)))
