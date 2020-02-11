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
    def get_base_model(cls, h:int,w:int,c:int):
        input_layer = tf.keras.Input(input_shape=(h,w,c))


        return model


class SoftmaxClassificationModel(ClasificationModel):

    @classmethod
    def get_classification_model(cls, base_model:tf.keras.Model,classes:int)->tf.keras.Model:
        x = tf.keras.layers.Flatten()(model.output)
        x = tf.keras.Dense(activation='softmax')(x)
        