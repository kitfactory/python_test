import tensorflow_datasets as tfds
import tensorflow as tf

# tfds works in both Eager and Graph modes
tf.enable_eager_execution()

# See available datasets
print(tfds.list_builders())


builder = tfds.builder('mnist', data_dir='dataset_dir')
# builder.download_and_prepare()
