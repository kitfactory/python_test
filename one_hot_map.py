import tensorflow as tf


MIXUP_KEEPS = 10
CLASS = 2
MIXUP_HYPER_PARAM_ALPHA = 0.5

oh = tf.Variable([[1.0,0.0],[0,1.0],[0.5,0.5],[1.0,0.0],[0,1.0],[0.5,0.5],[1.0,0.0],[0,1.0],[0.5,0.5],[1.0,0.0],[0,1.0],[0.5,0.5],[1.0,0.0],[0,1.0],[0.5,0.5],[1.0,0.0],[0,1.0],[0.5,0.5]], dtype=tf.float32)
mixup_flag = tf.Variable(tf.zeros([MIXUP_KEEPS], dtype=tf.bool), dtype=tf.bool, use_resource=True)
mixup_label = tf.Variable(tf.zeros([MIXUP_KEEPS,CLASS], dtype=tf.float32), dtype=tf.float32, use_resource=True)

beta_dist = tf.distributions.Beta(MIXUP_HYPER_PARAM_ALPHA, MIXUP_HYPER_PARAM_ALPHA)

ds = tf.data.Dataset.from_tensor_slices(oh)

def mixup_true_func(idx, label):
    """mixupでキープしておいたラベルを返却する。
    
    Arguments:
        idx {integer} -- キープしていたラベルのインデックス
        label {tf.Tensor} -- 読み込んだデータのラベル値
    
    Returns:
        tf.Tensor -- インデックスに保持されていたOne-Hot済みのラベルの値
    """

    tf.Print(idx,[idx], 'In True Func')
    ret_label = tf.gather(mixup_label,indices=[[idx]])
    return ret_label

def mixup_false_func(idx, label):
    """mixupでラベル値をキープする。
    
    Arguments:
        idx {integer} -- キープして
        label {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    tf.Print(idx,[idx], 'In False Func')
    tf.scatter_update(mixup_flag,indices=[[idx]],updates=[[True]])
    return label

def map_fn(label):
    rnd = tf.random.uniform(shape=[1],minval=0,maxval=MIXUP_KEEPS,dtype=tf.int32)
    # tf.Print(rnd,[rnd], "random value :")
    idx = tf.reshape(rnd,[])
    tf.Print(idx,[idx], "idx")
    z = tf.gather(mixup_flag,[[idx]])
    z = tf.reshape(z,[])
    # tf.Print( z,[z], "mixup_flag : ")

    label2 = tf.cond(z,true_fn=lambda: mixup_true_func(idx,label),false_fn=lambda: mixup_false_func(idx,label))
    with tf.control_dependencies([idx,label2]):
        tf.scatter_update(mixup_label,indices=[[idx]], updates=[[label]])

    lambda_param = beta_dist.sample()
    with tf.control_dependencies([label2,lambda_param]):
        new_label = (lambda_param * label) + (1-lambda_param)*label2
    return new_label, label, label2, lambda_param

ds = ds.map(map_fn)

iterator = ds.make_initializable_iterator()
elem = iterator.get_next()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer)
    try:
        while True:
            new_label, label, label2, lambda_param = sess.run(elem)
            print('result=' , new_label, ' label=',label, ' label2=',label2 , 'lambda=',lambda_param)
    except tf.errors.OutOfRangeError as err:
        print('End')
