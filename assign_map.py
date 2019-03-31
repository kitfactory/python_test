import tensorflow as tf

ds = tf.data.Dataset.from_tensor_slices([1,2,3,4,5,6,7,8,9,10])

v = tf.Variable(0, dtype=tf.int32, use_resource=True)

def map_fn(x):
    s = x + v
    tf.assign(v , s)
    return x

ds = ds.map(map_fn)
iterator = ds.make_initializable_iterator()
next = iterator.get_next()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('z:' , sess.run(z))
    sess.run(iterator.initializer)
    try:
        while True:
            x = sess.run(next)
            print('x:' , x)
    except tf.errors.OutOfRangeError as err:
        print('v:' , sess.run(v))
        print('end')
