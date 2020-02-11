import tensorflow as tf
from typing import List
from dataset import CatsVsDogsDataset

class Trainer():

    def __init__(model:tf.kera.Model,train:tf.data.Dataset,train_size:int, validation:tf.data.Dataset,validation_size:int):
        self.model = model
        self.train = train
        self.train_size = train_size
        self.validation = validation
        self.validation_size = validation_size

    def train(self, batch_size=10,epochs=100,logdir="./logs", loss="categorical_crossentropy",optimizer="adam"):
        train = train.repeat().batch(batch_size).prefetch(2)
        validation = validation.repeat().batch(batch_size).prefetch(2)
        callback:List[tf.keras.callbacks.Callback] = [
            tf.keras.callbacks.TensorBoard(log_dir=logdir),
            tf.keras.callbacks.ModelCheckpoint(filepath="model.hdf5",save_best_only=True,monitor="val_acc")
        ]
        model.compile(loss=loss,optimizer=optimizer,metrics="accuracy")
        model.summary()

        model.fit(
            x=train,
            steps_per_epoch=self.train_size//batch_size,
            validation_data=validation,
            validation_steps=self.validation_size//batch_size,
            batch_size=epochs
        )

if __name__ =="__main__":
    train = CatsVsDogsDataset.get_train_set()
    validation = CatsVsDogsDataset.get_validation_set()
    train_num, validation_num, _  = CatsVsDogsDataset.get_data_size()
    