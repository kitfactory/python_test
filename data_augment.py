import tensorflow as tf
import tensorflow_probability as tfp

from dataset import Cifar10Dataset
from dataset import DatasetUtil
from typing import Callable
from typing import Dict


class DataAugument():

    @classmethod
    def mixup_apply(cls, dataset:tf.data.Dataset, mixup_size:int, alpha: float)->Callable[[tf.data.Dataset],tf.data.Dataset]:
        """Mixup拡張をおこなう。ラベルはone-hot化されている事。
           dataset:tf.data.Dataset = somedataset
           mixuped = dataset.apply(DataAugument.mixup_apply(100,0.8))

        Args:
            dataset (tf.data.Dataset): [description]
            mix_size (int): [description]
            alpha (float): [description]
        
        Returns:
            tf.data.Dataset: [description]
        """

        def mixup(cls,dataset:tf.data.Dataset)->tf.data.Dataset:
            shuffle_dataset = dataset.shuffle(mix_size)
            zipped = tf.data.Dataset.zip((dataset,shuffle_dataset))

            def __mixup_map(data,shuffled):
                print(data)
                print(shuffled)
                dist = tfp.distributions.Beta([alpha],[alpha])
                beta = dist.sample([1])[0][0]
        
                ret = {}
                ret["image"] = (data["image"]) * beta + (shuffled["image"] * (1-beta)),
                ret["label"] = (data["label"]) * beta + (shuffled["label"] * (1-beta)),
                return ret
        
            return zipped.map(__mixup_map,num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return mixup
    
    

    
if __name__ == '__main__':
    train = Cifar10Dataset.get_train_set()
    train = train.map( DatasetUtil.image_classification_util(classes=10),num_parallel_calls=tf.data.experimental.AUTOTUNE )
    mixup = train.apply(DataAugument.mixup(mixup_size=100,alpha=0.8))
    for d in mixup:
        print(d["label"])
