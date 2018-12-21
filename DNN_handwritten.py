import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

# requirements of my fking env
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# load data from MNIST
data_dir = '/temp/mnist_data'
data = input_data.read_data_sets(data_dir, one_hot=True)

# define a layer function for later use
def add_layer(inputs, in_size, out_size, activation_function=None, name="hidden_layer"):
    with tf.name_scope(name):
        W = tf.Variable(tf.random_normal([in_size, out_size]))
        b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        Wx_plus_b = tf.matmul(inputs, W) + b
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

# initializing constants of network
num_steps = 30000
minibatch_size = 100
keep_prob = 1.0


# building network: a 3-layer DNN
# input layer: 784 neurons
x = tf.placeholder(tf.float32, shape=(None, 784), name="x")

# layer 1: 16 neurons, ReLU
layer1_outputs = add_layer(x, 784, 128, activation_function=tf.nn.relu, name="layer_1")
layer1_outputs = tf.nn.dropout(layer1_outputs, keep_prob)

# layer 2: 16 neurons, ReLU
layer2_outputs = add_layer(layer1_outputs, 128, 64, activation_function=tf.nn.relu, name="layer_2")
layer2_outputs = tf.nn.dropout(layer2_outputs, keep_prob)

# output layer: 10 neurons, softmax
y_pred = add_layer(layer2_outputs, 64, 10, name="output_layer")


# calculate loss: cross entropy
y_true = tf.placeholder(tf.float32, [None, 10])
loss_matrix = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
loss = tf.reduce_mean(loss_matrix)

#learning rate
learning_rate = tf.Variable(0.2)

# using Adam optimier to minimize loss
gd_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

# accuracy check
correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))


with tf.Session() as sess:

    # save logs
    tf.summary.FileWriter("./log", tf.get_default_graph())

    # training
    sess.run(tf.global_variables_initializer())
    for step in range(num_steps):
        batch_xs, batch_ys = data.train.next_batch(minibatch_size)
        sess.run(gd_step, feed_dict={x: batch_xs, y_true: batch_ys})


    # accuracy check; testing
        if step % 1000 == 0:
            ans = sess.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels})
            print("Accuracy: {:.4}%".format(ans*100))
