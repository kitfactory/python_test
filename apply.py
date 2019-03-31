import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

tf.logging.set_verbosity(tf.logging.ERROR)

d = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2)]
ds = tf.data.Dataset.from_tensor_slices(d)

def to_class(d):
    x = tf.gather(d,indices=[0],axis=0)
    x = tf.reshape(x, shape=[] )
    return x

ds = ds.apply(tf.data.experimental.rejection_resample(to_class,target_dist=[0.1,0.1,0.8],initial_dist=[0.33,0.34,0.33]))



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    iterator = ds.make_one_shot_iterator()
    next = iterator.get_next()
    try:
        while True:
            print('Got !!' , sess.run(next))
    except tf.errors.OutOfRangeError as error:
        print('end')
