import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_dir = '/temp/mnist_data'
num_steps = 1000
minibatch_size = 100

# load data from MNIST into data_dir directory
data = input_data.read_data_sets(data_dir, one_hot=True)
# one_hot: [0,1,0,0,0,0,0,0,0]. easier to compute loss func later

x = tf.placeholder(tf.float32, shape=(None, 784))
# tf.placeholder(dtype, shape=None, name=None)
# [None, 784]表示列是784，行不定
# 所以placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。
# 等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。

W = tf.Variable(tf.zeros([784, 10]))
# tf.Variable.init(initial_value, trainable=True, collections=None, validate_shape=True, name=None)
# use [] or () to denote shape

y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)

loss_matrix = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
# tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
# logits：就是神经网络最后一层的输出
# 第二个参数labels：实际的标签

cross_entropy = tf.reduce_mean(loss_matrix)
# 求平均值tf.reduce_mean(input_tensor, reduction_indices=None, keep_dims=False, name=None)
# 参数1--input_tensor:待求值的tensor
# 参数2--reduction_indices:在哪一维上求解

gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# tf.train.GradientDescentOptimizer is a subclass of tf.train.Optimizer 类似的子类还有AdagradOptimizer或者MomentumOptimizer
# 0.5: learning rate
# tf.train.Optimizer.minimize(loss, global_step=None, var_list=None, ...)
# - loss: A Tensor containing the value to minimize.
# - global_step: Optional Variable to auto_increment by one after the variables have been updated.
# - var_list: Optional list of Variable objects to update to minimize loss. Defaults to the list of variables collected in the graph under the key GraphKeys.TRAINABLE_VARIABLES.

correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
# tf.equal(x, y, name=None) 功能：对比两个矩阵/向量的元素是否相等，如果相等就返回True，反之返回False。
# tf.argmax(vector, axis)：按照axis的维度比较，返回的是vector中的最大值的索引号，如果vector是一个向量，那就返回一个值，如果是一个矩阵，那就返回一个向量，这个向量的每一个维度都是相对应矩阵行的最大值元素的索引号。

accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))
# tf.cast(x, dtype, name=None) 将x的数据格式转化成dtype

with tf.Session() as sess:

    # Train
    sess.run(tf.global_variables_initializer())
    for _ in range(num_steps):
        batch_xs, batch_ys = data.train.next_batch(minibatch_size)
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})
        # sess.run(fetches, feed_dict=None, options=None, run_metadata=None)

    # Test
    ans = sess.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels})

print("Accuracy: {:.4}%".format(ans*100))
