import tensorflow as tf
import numpy as np
import optuna

import tensorflow.keras.backend as K

keras = tf.keras

class OptunaCallback(keras.callbacks.Callback):
    def __init__(self, trial):
        self.trial = trial

    def on_epoch_end(self, epoch, logs):
        current_val_error = 1.0 - logs["val_acc"]
        self.trial.report(current_val_error, step=epoch)
        # 打ち切り判定
        if self.trial.should_prune(epoch):
            raise optuna.structs.TrialPruned()

def create_dataset(batch_size):
    (x_train,y_train), (x_valid,y_valid)=keras.datasets.fashion_mnist.load_data()
    x_train = x_train / 255.0
    x_train = np.reshape(x_train,(60000,28,28,1))
    x_train = x_train.astype(np.float32)
    y_train = tf.keras.utils.to_categorical(y_train,num_classes=10)

    x_valid = x_valid / 255.0
    x_valid = np.reshape(x_valid, (10000,28,28,1))
    x_valid = x_valid.astype(np.float32)
    y_valid = tf.keras.utils.to_categorical(y_valid,num_classes=10)

    c = tf.Variable(0, dtype=tf.int32)

    def map_fn(x,y):
        # c.assign_add(1)
        return (x,y)

    train_set= tf.data.Dataset.from_tensor_slices((x_train,y_train)).map(map_func=map_fn).repeat().batch(batch_size)
    valid_set = tf.data.Dataset.from_tensor_slices((x_valid,y_valid)).repeat().batch(batch_size)
    return (train_set, 60000),(valid_set, 10000)


def create_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(28,28,1)))
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    return model


def train(lr, max_epoch, trial):
    batch_size = 100
    (train_set, train_num), (valid_set, valid_num) = create_dataset(batch_size)

    print(train_set)

    model = create_model()
    optimizer = tf.keras.optimizers.SGD(lr=lr)
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
    model.summary()
    optuna_callback = OptunaCallback(trial)
    model.fit(train_set,steps_per_epoch=train_num/batch_size,validation_data=valid_set,validation_steps=valid_num/batch_size,epochs=max_epoch)

def objective(trial):
    max_epoch = 20
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 1e0)
    K.clear_session()
    train(learning_rate, max_epoch, trial)
    hist = train(learning_rate, max_epoch, trial)
    return 1.0 - np.max(hist["val_acc"])

if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=2)

