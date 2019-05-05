from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

from typing import ClassVar

import os
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)
sess = tf.Session()
sess.as_default()

dtype='float16'
tf.keras.backend.set_floatx('float16')
# default is 1e-7 which is too small for float16.  Without adjusting the epsilon, we will get NaN predictions because of divide by zero problems
tf.keras.backend.set_epsilon(1e-4) 


class Mixup():

    def __init__(self):
        super().__init__()

    def config(self, keeps: int, classes: int, img_size:int, channel:int , alpha: float):
        self.mixup_keeps = keeps
        self.num_class = classes
        self.alpha = alpha
        self.mixup_flag = tf.Variable(tf.zeros([self.mixup_keeps], dtype=tf.bool), dtype=tf.bool, use_resource=True)
        self.mixup_label = tf.Variable(tf.zeros([self.mixup_keeps, self.num_class], dtype=tf.float32), dtype=tf.float32, use_resource=True)
        self.mixup_img = tf.Variable(tf.zeros([self.mixup_keeps,img_size, img_size, channel], dtype=tf.float32), dtype=tf.float32, use_resource= True)
        self.beta_dist = tf.distributions.Beta(self.alpha, self.alpha)

    def mixup_dataset(self, dataset:tf.data.Dataset):

        # def mixup_true_func(self, idx, img, label):
        def mixup_true_func(idx, img, label):
            """mixupでキープしておいたラベルを返却する。

            Arguments:
                idx {integer} -- キープしていたラベルのインデックス
               label {tf.Tensor} -- 読み込んだデータのラベル値
    
            Returns:
                tf.Tensor -- インデックスに保持されていたOne-Hot済みのラベルの値
            """
            # tf.Print(idx,[idx], 'In True Func')
            ret_img = tf.gather(self.mixup_img, indices=[[idx]])
            ret_label = tf.gather(self.mixup_label,indices=[[idx]])
            return ret_img, ret_label

        # def mixup_false_func(self, idx, img, label):
        def mixup_false_func(idx, img, label):
            """mixupでラベル値をキープする。

            Arguments:
                idx {integer} -- キープして
                label {[type]} -- [description]
    
            Returns:
                [type] -- [description]
            """
            # tf.Print(idx,[idx], 'In False Func')
            tf.scatter_update(self.mixup_flag,indices=[[idx]],updates=[[True]])
            return img, label
    
        def map_func(img, label):
            rnd = tf.random.uniform(shape=[1],minval=0,maxval=self.mixup_keeps,dtype=tf.int32)
            idx = tf.reshape(rnd,[])
            z = tf.gather(self.mixup_flag,[[idx]])
            z = tf.reshape(z,[])

            img2, label2 = tf.cond(z,true_fn=lambda: mixup_true_func(idx,img, label), false_fn=lambda: mixup_false_func(idx,img, label))

            with tf.control_dependencies([idx,label2]):
                tf.scatter_update(self.mixup_img,indices=[[idx]], updates=[[img2]])
                tf.scatter_update(self.mixup_label,indices=[[idx]], updates=[[label2]])

            lambda_param = self.beta_dist.sample()
            with tf.control_dependencies([label2,lambda_param]):
                new_img = (lambda_param*img) + (1-lambda_param)*img2
                new_label = (lambda_param * label) + (1-lambda_param)*label2
                return new_img, new_label

        return dataset.map(map_func=map_func)



(x_train, y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()

x_train = x_train / 255
# x_train = x_train.astype(np.float32)
x_train = x_train.astype(np.float16)

# y_train = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype='float16')
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

z = tf.zeros([10], dtype=tf.bool)

mixup_flag = tf.Variable( tf.zeros(shape=[10],dtype=tf.bool), dtype=tf.bool, use_resource=True)
# mixup_label = tf.Variable(shape=[10,10], dtype=tf.float32, use_resource=True)
# mixup_img_memo = tf.get_variable("img_memo", [10,28,28,1], dtype=tf.float32, initializer=tf.zeros_initializer)

iz = np.zeros([28,28],dtype=np.float32)
# mixup_img_memo = tf.Variable(iz, dtype=tf.float32,name='img_memo')
mixup_img_memo = tf.Variable(iz, dtype=tf.float16,name='img_memo')
counter = tf.Variable(0, dtype=tf.int32)

# mixup_label_memo = tf.Variable(tf.zeros([1,10], dtype=tf.float32))
# mix-up randomをだして、そのidに入れる。


def map_fn(x,y):
    x = tf.reshape(x,(-1,28,28,1))
    return x, y


# with tf.device('/cpu:0'):
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# mixup = Mixup()
# mixup.config(keeps=5, classes=10, img_size=28, channel=1, alpha=0.5)
# dataset = mixup.mixup_dataset(dataset)
dataset = dataset.batch(100)
dataset = dataset.repeat() # .map(map_fn,num_parallel_calls=4)

print(dataset)


# dataset = dataset.map(mix_up,num_parallel_calls=4).batch(100).map(map_fn,num_parallel_calls=4)
# dataset = dataset.batch(100).map(map_fn,num_parallel_calls=4)
# dataset = dataset.prefetch(1)

# iterator = dataset.make_initializable_iterator()
# init = iterator.initializer
# next = iterator.get_next()

# sess.run(tf.global_variables_initializer())
# sess.run(iterator.initializer)

# try:
#    while True:
        # img, label = sess.run(next)
#         img, label = sess.run(next)
# except tf.errors.OutOfRangeError as err:
#    print('End')
# 
# exit()

'''
# iterator = dataset.make_one_shot_iterator()
# x, y = iterator.get_next()
# print(x,y)
# print(mix_up_label)
print(r)
print(counter)

x, y = iterator.get_next()
# print(x,y)
# print(mix_up_label)
print(r)
print(counter)

# print(dataset)
# print(x)
exit()
'''
# print(dataset)




model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='SGD',
              metrics=['accuracy'])

model.summary()

sess.run(tf.global_variables_initializer())
# print( 'flag : ', sess.run(flg))

# history = model.fit(x_train, y_train,
#                    batch_size=50,
#                    steps_per_epoch=120,
#                    epochs=10,
#                    verbose=1)

checkpoint = tf.keras.callbacks.ModelCheckpoint('model', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)

history = model.fit(dataset,
                    steps_per_epoch=600,
                    epochs=10,
                    verbose=1)


sess.close()