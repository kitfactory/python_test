import tensorflow as tf

class ImageAugument():

    def __init__(self):
        super(self).__init__()

    def resize(self, dataset:tf.data.Dataset, target_height:int, target_width:int ):
        
        def resize_map_fn(x, y):
            x = tf.image.resize_image_with_pad( x, target_height, target_width)
            return x, y
        return dataset.map(resize_map_fn)
