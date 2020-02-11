import tensorflow as tf

from abc import ABC, abstractmethod

class ImageBaseModel(ABC):
    @classmethod
    @abstractmethod
    def get_base_model(cls, h:int,w:int,c:int):
        """ベースとなるモデルを提供する。
        
        Args:
            ImageBaseModel ([type]): [description]
            h (int): 入力サイズ hight
            w (int): 入力サイズ width
            c (int): 入力サイズ channel
        
        Returns:
            [type]: CNNモデル
        """
        raise NotImplementedError()

class ClasificationModel(ABC):
    @classmethod
    @abstractmethod
    def get_classification_model(cls, base_model:tf.keras.Model)->tf.keras.Model:
        """ ベースモデルに分類部分をつけた分類用のモデルを提供する。
        
        Args:
            base_model (tf.keras.Model): ベースモデル
            classes (int): 分類数
        
        Returns:
            tf.keras.Model: 分類モデル
        """
        raise NotImplementedError()

class SimpleCNN(ImageBaseModel):
    @classmethod
    def get_base_model(cls, h:int,w:int,c:int):
        """単純なCNNモデルを提供する。
        
        Args:
            ImageBaseModel ([type]): [description]
            h (int): 入力サイズ hight
            w (int): 入力サイズ width
            c (int): 入力サイズ channel
        
        Returns:
            [type]: CNNモデル
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

class ResNet50(ImageBaseModel):
    @classmethod
    def get_base_model(cls, h:int,w:int,c:int, weights:bool=False):
        """ResNet CNNモデルを提供する。
        
        Args:
            ImageBaseModel ([type]): [description]
            h (int): 入力サイズ hight
            w (int): 入力サイズ width
            c (int): 入力サイズ channel
        
        Returns:
            [type]: CNNモデル
        """
        if weights==False:
            w_imagenet = None
        else:
            w_imagenet = "imagenet"
        model = tf.keras.applications.ResNet50(include_top=False,input_shape=(h,w,c),weights=w_imagenet)
        return model

class SimpleSoftmaxClassificationModel(ClasificationModel):

    @classmethod
    def get_classification_model(cls, base_model:tf.keras.Model,classes:int)->tf.keras.Model:
        """ ベースモデルに分類部分をつけた分類用のモデルを提供する。
        
        Args:
            base_model (tf.keras.Model): ベースモデル
            classes (int): 分類数
        
        Returns:
            tf.keras.Model: 分類モデル
        """
        x = tf.keras.layers.Flatten()(base_model.output)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dense(100)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Dense(classes, activation='softmax')(x)
        return tf.keras.Model(base_model.input, x)

if __name__ == '__main__':
    base = SimpleCNN.get_base_model(28,28,1)
    model = SimpleSoftmaxClassificationModel.get_classification_model(base,10)
    res_base = ResNet50.get_base_model(224,224,3)
