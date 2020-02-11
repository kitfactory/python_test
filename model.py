import tensorflow as tf

from abc import ABC, abstractmethod

class ImageBaseModel(ABC):
    @classmethod
    @abstractmethod
    def get_base_model(cls, h:int,w:int,c:int):
        raise NotImplementedError()

class ClasificationModel(ABC):
    @classmethod
    @abstractmethod
    def get_classification_model(cls, base_model:tf.keras.Model)->tf.keras.Model:
        raise NotImplementedError()

class SimpleCNN(ImageBaseModel):
    @classmethod
    def get_base_model(cls, h:int,w:int,c:int):
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32,(3,3),padding='same',input_shape=(h,w,c)))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Conv2D(32,(3,3),padding='same'))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
        model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Conv2D(64,(3,3),padding='same'))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Conv2D(64,(3,3),padding='same'))
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))
        model.add(tf.keras.layers.Dropout(0.25))
        return model

        """
        input_layer = tf.keras.Input(shape=(h,w,c),name='input',dtype=tf.float32)
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', name='conv2d-1')(input_layer)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', name='conv2d-2')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),name='maxpool-1')(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', name='conv2d-3')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', name='conv2d-4')(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2),name='maxpool-2')(x)

        return tf.keras.Model(input_layer,x)


class SimpleSoftmaxClassificationModel(ClasificationModel):

    @classmethod
    def get_classification_model(cls, base_model:tf.keras.Model,classes:int)->tf.keras.Model:
        x = tf.keras.layers.Flatten()(base_model.output)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dense(100)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dense(classes, activation='softmax')(x)
        return tf.keras.Model(base_model.input, x)

if __name__ == '__main__':
    base = SimpleCNN.get_base_model(28,28,1)
    model = SimpleSoftmaxClassificationModel.get_classification_model(base,10)
