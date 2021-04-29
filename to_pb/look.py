import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

g = tf.Graph()

with g.as_default() as g:
    tf.train.import_meta_graph('./summary/20210428/models/hed-model-10000.meta')

with tf.Session(graph=g) as sess:
    file_writer = tf.summary.FileWriter(logdir='ckpt_log/', graph=g)

