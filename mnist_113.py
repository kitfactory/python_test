from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

# tf.enable_eager_execution()


# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)
sess = tf.Session()
sess.as_default()

class Mixup():

    def __init__(self, keeps: int, classes: int, img_size:int, channel:int , alpha: float):
        """初期化をする。
        
        Arguments:
            keeps {int} -- MIXするために保持しておく数
            classes {int} -- ラベルのクラス数
            alpha {float} -- mixupするときのハイパーパラメータ
        """
        self.mixup_keeps = keeps
        self.num_class = classes
        self.alpha = alpha
        self.mixup_flag = tf.Variable(tf.zeros([self.mixup_keeps], dtype=tf.bool), dtype=tf.bool, use_resource=True)
        self.mixup_label = tf.Variable(tf.zeros([self.mixup_keeps, self.num_class], dtype=tf.float32), dtype=tf.float32, use_resource=True)
        # self.mixup_img = tf.Variable(tf.zeros([self.mixup_keeps,img_size, img_size, channel], dtype=tf.float32), dtype=tf.float32, use_resource= True)
        self.beta_dist = tf.distributions.Beta(alpha, alpha)

    # def mixup_true_func(self, idx, img, label):
    def mixup_true_func(self, idx, label):
        """mixupでキープしておいたラベルを返却する。
    
        Arguments:
            idx {integer} -- キープしていたラベルのインデックス
            label {tf.Tensor} -- 読み込んだデータのラベル値
    
        Returns:
            tf.Tensor -- インデックスに保持されていたOne-Hot済みのラベルの値
        """
        tf.Print(idx,[idx], 'In True Func')
        # ret_img = tf.gather(mixup_img, indices=[[idx]])
        ret_label = tf.gather(mixup_label,indices=[[idx]])
        # return ret_img, ret_label
        return ret_label


    # def mixup_false_func(self, idx, img, label):
    def mixup_false_func(self, idx, label):
        """mixupでラベル値をキープする。
    
        Arguments:
            idx {integer} -- キープして
            label {[type]} -- [description]
    
        Returns:
            [type] -- [description]
        """
        tf.Print(idx,[idx], 'In False Func')
        tf.scatter_update(mixup_flag,indices=[[idx]],updates=[[True]])
        # return img, label
        return label

    def map_func(self, label):
        rnd = tf.random.uniform(shape=[1],minval=0,maxval=self.num_class,dtype=tf.int32)
        idx = tf.reshape(rnd,[])
        z = tf.gather(mixup_flag,[[idx]])
        z = tf.reshape(z,[])

        img2, label2 = tf.cond(z,true_fn=lambda: self.mixup_true_func(idx,label),false_fn=lambda: self.mixup_false_func(idx,label))

        with tf.control_dependencies([idx,label2]):
            tf.scatter_update(mixup_label,indices=[[idx]], updates=[[label]])
        lambda_param = beta_dist.sample()
        with tf.control_dependencies([label2,lambda_param]):
            # new_img = (lambda_param*img) + (1-lambda_param)*img2
            new_label = (lambda_param * label) + (1-lambda_param)*label2
    
        return new_label


# tf.
# config.gpu_options.allow_growth = True
# config = tf.ConfigProto()


# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)


(x_train, y_train),(x_test,y_test)=tf.keras.datasets.mnist.load_data()

x_train = x_train / 255
x_train = x_train.astype(np.float32)

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

z = tf.zeros([10], dtype=tf.bool)

mixup_flag = tf.Variable( tf.zeros(shape=[10],dtype=tf.bool), dtype=tf.bool, use_resource=True)
# mixup_label = tf.Variable(shape=[10,10], dtype=tf.float32, use_resource=True)
# mixup_img_memo = tf.get_variable("img_memo", [10,28,28,1], dtype=tf.float32, initializer=tf.zeros_initializer)

iz = np.zeros([28,28],dtype=np.float32)
mixup_img_memo = tf.Variable(iz, dtype=tf.float32,name='img_memo')
counter = tf.Variable(0, dtype=tf.int32)

# mixup_label_memo = tf.Variable(tf.zeros([1,10], dtype=tf.float32))
# mix-up randomをだして、そのidに入れる。

r = tf.Variable(0, tf.int32)
print('r', r)

hoge = tf.assign(r, 10)
print('r', r)


mixup = MixupAugmentation(keeps=5, classes=10, img_size=28, alpha=0.5)


def mix_up(x, y):
    rnd = tf.random.uniform(shape=[1], dtype=tf.int32,minval=0,maxval=10)
    ridx = tf.reshape(rnd,[])
    flgs = mixup_flag
    flg = tf.gather(flgs, indices=[[5]])
    return x, y

    # c = counter + 1
    # tf.assign(counter, c)
    # with tf.control_dependencies([ridx]):
    # tf.scatter_update(mixup_img_memo,indices=[[0]],updates=[x])]
    # print(mixup_img_memo)
    # print(x)
    # tf.assign(mixup_img_memo,img)

def map_fn(x,y):
    x = tf.reshape(x,(-1,28,28,1))
    return x, y


# with tf.device('/cpu:0'):
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.batch(100)


dataset = dataset.repeat(1) # .map(map_fn,num_parallel_calls=4)

print(dataset)


# dataset = dataset.map(mix_up,num_parallel_calls=4).batch(100).map(map_fn,num_parallel_calls=4)
# dataset = dataset.batch(100).map(map_fn,num_parallel_calls=4)
# dataset = dataset.prefetch(1)

# iterator = dataset.make_initializable_iterator()
# init = iterator.initializer
# next = iterator.get_next()

# sess.run(tf.global_variables_initializer())
# sess.run(iterator.initializer)


# c = 0
# while True:
#     c = c + 1
#     img, label = sess.run(next)
    # print(c,label)

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


history = model.fit(dataset,
                    steps_per_epoch=600,
                    epochs=10,
                    verbose=1)


sess.close()