from typing import List
from enum import Enum
from pymongo import MongoClient
import tensorflow as tf

class PyMongType(Enum):
    INT32 = 0,
    FLOAT32 = 1,
    STRING =2

class PyMongoAttribute():
    def __init__(self, attribute:str, type_of_attr:PyMongType):
        self.attribute = attribute
        self.type = type_of_attr


class PyMongoDataset():

    def __init__(self, host:str, port:int, db:str, collection:str, attributes:List[PyMongoAttribute]):
        # mongodb へのアクセスを確立
        self.client = MongoClient(host, port)
        self.db = self.client[db]
        self.co = self.db[collection]
        self.attributes = attributes

    def get_generator(self):
        for data in self.co.find():
            t = []
            for attr in self.attributes:
                if attr.type == PyMongType.INT32:
                    t.append(int(data[attr.attribute]))
                elif attr.type == PyMongType.FLOAT32:
                    t.append(float(data[attr.attribute]))
                else:
                    t.append(data[attr.attribute])
            yield tuple(t)
    
    def create_dataset(self)->tf.data.Dataset:
        t = []
        for a in self.attributes:
            if a.type == PyMongType.INT32:
                t.append(tf.int32)
            elif a.type == PyMongType.FLOAT32:
                t.append(tf.float32)
            else:
                t.append(tf.string)
        o = tuple(t)
        return tf.data.Dataset.from_generator(self.get_generator,output_types=o)


if __name__ == '__main__':

    attributes = [
        PyMongoAttribute("SepalLength",PyMongType.FLOAT32),
        PyMongoAttribute("SepalWidth",PyMongType.FLOAT32)
    ]

    gen = PyMongoDataset('localhost', 27017, 'iris', 'iris_collection', attributes)
    ds = gen.get_dataset()
    iterator = ds.make_initializable_iterator()
    init = iterator.initializer
    next = iterator.get_next()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(init)
        while True:
            data = sess.run(next)
            print(data)
               
