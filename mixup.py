import tensorflow as tf

class Mixup():

    def __init__(self):
        super().__init__()

    def config(self, keeps: int, classes: int, img_size:int, channel:int , alpha: float):
        self.mixup_keeps = keeps
        self.num_class = classes
        self.alpha = alpha
        self.mixup_flag = tf.Variable(tf.zeros([self.mixup_keeps], dtype=tf.bool), dtype=tf.bool, use_resource=True)
        self.mixup_label = tf.Variable(tf.zeros([self.mixup_keeps, self.num_class], dtype=tf.float32), dtype=tf.float32, use_resource=True)
        self.mixup_img = tf.Variable(tf.zeros([self.mixup_keeps,img_size, img_size, channel], dtype=tf.float32), dtype=tf.float32, use_resource= True)
        self.beta_dist = tf.distributions.Beta(self.alpha, self.alpha)

    def mixup_dataset(self, dataset:tf.data.Dataset):

        # def mixup_true_func(self, idx, img, label):
        def mixup_true_func(idx, img, label):
            """mixupでキープしておいたラベルを返却する。

            Arguments:
                idx {integer} -- キープしていたラベルのインデックス
               label {tf.Tensor} -- 読み込んだデータのラベル値
    
            Returns:
                tf.Tensor -- インデックスに保持されていたOne-Hot済みのラベルの値
            """
            # tf.Print(idx,[idx], 'In True Func')
            ret_img = tf.gather(self.mixup_img, indices=[[idx]])
            ret_label = tf.gather(self.mixup_label,indices=[[idx]])
            return ret_img, ret_label

        # def mixup_false_func(self, idx, img, label):
        def mixup_false_func(idx, img, label):
            """mixupでラベル値をキープする。

            Arguments:
                idx {integer} -- キープして
                label {[type]} -- [description]
    
            Returns:
                [type] -- [description]
            """
            # tf.Print(idx,[idx], 'In False Func')
            tf.scatter_update(self.mixup_flag,indices=[[idx]],updates=[[True]])
            return img, label
    
        def map_func(img, label):
            rnd = tf.random.uniform(shape=[1],minval=0,maxval=self.mixup_keeps,dtype=tf.int32)
            idx = tf.reshape(rnd,[])
            z = tf.gather(self.mixup_flag,[[idx]])
            z = tf.reshape(z,[])

            img2, label2 = tf.cond(z,true_fn=lambda: mixup_true_func(idx,img, label), false_fn=lambda: mixup_false_func(idx,img, label))

            with tf.control_dependencies([idx,label2]):
                tf.scatter_update(self.mixup_img,indices=[[idx]], updates=[[img2]])
                tf.scatter_update(self.mixup_label,indices=[[idx]], updates=[[label2]])

            lambda_param = self.beta_dist.sample()
            with tf.control_dependencies([label2,lambda_param]):
                new_img = (lambda_param*img) + (1-lambda_param)*img2
                new_label = (lambda_param * label) + (1-lambda_param)*label2
                return new_img, new_label

        return dataset.map(map_func=map_func)
