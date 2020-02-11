import tensorflow as tf
from typing import List


class Trainer():

    def __init__(self, model:tf.keras.Model, train:tf.data.Dataset,train_size:int, validation:tf.data.Dataset,validation_size:int):
        self.model = model
        self.train_dataset = train
        self.train_size = train_size
        self.validation_dataset = validation
        self.validation_size = validation_size

    def train(self, batch_size=10,epochs=100,logdir="./logs", loss="categorical_crossentropy",optimizer="adam"):
        print("train method")
        self.train_dataset = self.train_dataset.repeat().batch(batch_size).prefetch(2)
        self.validation_dataset = self.validation_dataset.repeat().batch(batch_size).prefetch(2)

        callback:List[tf.keras.callbacks.Callback] = [
            tf.keras.callbacks.TensorBoard(log_dir=logdir),
            tf.keras.callbacks.ModelCheckpoint(filepath="model.hdf5",save_best_only=True,monitor="val_acc")
        ]
        print("compile")
        self.model.compile(loss=loss,optimizer=optimizer,metrics=["acc"])
        print("summary")
        self.model.summary()

        print(self.train_dataset)

        print("fit")
        self.model.fit(
            x=self.train_dataset,
            steps_per_epoch=self.train_size//batch_size,
            validation_data=self.validation_dataset,
            validation_steps=self.validation_size//batch_size,
            epochs=epochs
        )
#            callbacks=callback
#        )

if __name__ =="__main__":
    from dataset import MnistDataset
    from dataset import DatasetUtil
    from model import SimpleCNN
    from model import SimpleSoftmaxClassificationModel

    train = MnistDataset.get_train_set().map(DatasetUtil.image_classification_util(10))
    validation = MnistDataset.get_validation_set().map(DatasetUtil.image_classification_util(10))
    train_size, validation_size, _  = MnistDataset.get_data_size()
    base = SimpleCNN.get_base_model(28,28,1)
    softmax_model = SimpleSoftmaxClassificationModel.get_classification_model(base,10)
    trainer:Trainer = Trainer(
        softmax_model,
        train=train,
        train_size=train_size,
        validation=validation,
        validation_size=validation_size
    )
    trainer.train()