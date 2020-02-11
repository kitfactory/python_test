from abc import ABC, abstractmethod

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets import Split
from typing import Dict

class ImageClassifyDataset(ABC):

    @classmethod
    @abstractmethod
    def get_train_set(cls)->tf.data.Dataset:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_validation_set(cls)->tf.data.Dataset:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_test_set(cls)->tf.data.Dataset:
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def get_data_size(cls)->(int,int,int):
        raise NotImplementedError()


class MnistDataset(ImageClassifyDataset):

    @classmethod
    def get_train_set(cls)->tf.data.Dataset:
        return tfds.load(name="mnist",split="train[:50000]")

    @classmethod
    def get_validation_set(cls)->tf.data.Dataset:
        return tfds.load("mnist",split="train[50000:]")

    @classmethod
    def get_test_set(cls)->tf.data.Dataset:
        return tfds.load("mnist",split="test")

    @classmethod
    def get_data_size(cls)->(int,int,int):
        return (50000,10000,10000)


class Cifar10Dataset(ImageClassifyDataset):

    @classmethod
    def get_train_set(cls)->tf.data.Dataset:
        return tfds.load(name="cifar10",split="train[:50000]")

    @classmethod
    def get_validation_set(cls)->tf.data.Dataset:
        return tfds.load("cifar10",split="train[50000:]")

    @classmethod
    def get_test_set(cls)->tf.data.Dataset:
        return tfds.load("cifar10",split="test")

    @classmethod
    def get_data_size(cls)->(int,int,int):
        return (50000,10000,10000)


class CatsVsDogsDataset(ImageClassifyDataset):
    @classmethod
    def get_train_set(cls)->tf.data.Dataset:
        return tfds.load(name="cats_vs_dogs",split="train[:20000]")

    @classmethod
    def get_validation_set(cls)->tf.data.Dataset:
        return tfds.load("cats_vs_dogs",split="train[20000:21600]")

    @classmethod
    def get_test_set(cls)->tf.data.Dataset:
        return tfds.load("cats_vs_dogs",split="train[21600:]")

    @classmethod
    def get_data_size(cls)->(int,int,int):
        return (20000,1600,1662)


class RockPaperScissorsDataset(ImageClassifyDataset):
    @classmethod
    def get_train_set(cls)->tf.data.Dataset:
        return tfds.load(name="rock_paper_scissors",split="train[:2320]")

    @classmethod
    def get_validation_set(cls)->tf.data.Dataset:
        return tfds.load("rock_paper_scissors",split="train[2320:]")

    @classmethod
    def get_test_set(cls)->tf.data.Dataset:
        return tfds.load("rock_paper_scissors",split="test")

    @classmethod
    def get_data_size(cls)->(int,int,int):
        return (2320,200,372)



class DatasetUtil():

    @classmethod
    def count_image_dataset(cls, dataset)->(int, Dict):
        total = 0
        count = {}
        for d in dataset:
            y = d["label"].numpy()
            c = count.get(y,0) + 1
            count[y] = c
            total = total + 1
        print(total)
        print(count)
        return total, count
    
    @classmethod
    def one_hot(cls, classes):
        def one_hot_map(data):
            data["label"] = tf.one_hot(data["label"], classes)
            return data
        return one_hot_map

    @classmethod
    def image_reguralization(cls, offset:float=0.0):
        def image_reguralization_map(data):
            data["image"]=tf.cast(data["image"], tf.float32)
            data["image"]=data["image"]/255.0 - offset
            return data
        return image_reguralization_map
    
    @classmethod
    def image_classification_util(cls, classes):
        def image_classificaiton_util_map(data:Dict)->Dict:
            data["label"] = tf.one_hot(data["label"], classes)
            data["image"] = tf.cast(data["image"], tf.float32)
            data["image"] = data["image"]/255.0
            return data
        return image_classificaiton_util_map
    
    @classmethod
    def resize_with_crop_or_pad(td.data.Dataset, hight:int, width;int)->Callable[[Dict]->Dict]:
        def __resize_with_crop_or_pad(data:Dict)->Dict:
            data["image"] = tf.image.resize_with_crop_or_pad(data["image"], hight, width)
            return data
        return __resize_with_crop_or_pad


if __name__ == "__main__":
    # train = MnistDataset.get_train_set()
    # DatasetUtil.count_image_dataset(train)

    # cifar10_train = Cifar10Dataset.get_train_set()
    # DatasetUtil.count_image_dataset(cifar10_train)

    cifar10_test = Cifar10Dataset.get_test_set()
    regularized_image = cifar10_test.map(DatasetUtil.image_classification_util(10))

