from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np


if tf.__version__.startswith('1.13'):
    tf.enable_eager_execution()
    print('EARGER MODE!!')



# np_val = np.array([[[1,2,3][4,5,6][7,8,9]],[[1,2,3][4,5,6][7,8,9]]])

np_val = np.array([[[1,2,3],[2,3,4],[3,4,5]],[[4,5,6],[5,6,7],[6,7,8]]])


x = tf.Variable(np_val, dtype=tf.float32)
update = tf.scatter_update(x,indices=[[0]],updates=[[[9,9,9],[9,9,9],[9,9,9]]])


print(update)
# x = tf.scatter_update(x,indices=[0],updates=[[1,1,1],[1,1,1],[1,1,1]])

print(tf.__version__)