import tensorflow as tf
import numpy as np

class ImageAugument():

    def __init__(self):
        super().__init__()

    def resize(self, dataset:tf.data.Dataset, target_height:int, target_width:int ):
        
        def resize_map_fn(x, y):
            x = tf.image.resize_image_with_pad( x, target_height, target_width)
            return x, y
        return dataset.map(resize_map_fn)


(x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255
x_train = x_train.astype(np.float32)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10, dtype='float32')

augment = ImageAugument()

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = augment.resize(dataset, 224, 224)
dataset = dataset.batch(100)
dataset = dataset.repeat() # .map(map_fn,num_parallel_calls=4)


class ResNet50():
    def __init__(self):

    def create_model(class_num: int):
        


tf.keras.application.

print(dataset)


