import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

import os

# requirements of my fking env
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#载入数据集

mnist1 = input_data.read_data_sets("MNIST_data", one_hot=True)



#每个批次大小

batch_size=100



#计算总共批次

n_batch = mnist1.train.num_examples // batch_size



#定义两个placeholder

x = tf.placeholder(tf.float32, [None,784])

y = tf.placeholder(tf.float32, [None,10])#输出只有10个

keep_prob = tf.placeholder(tf.float32)

lr = tf.Variable(0.001, dtype = tf.float32)



#创建简单神经网络

#W1 = tf.Variable(tf.zeros([784,200]))#784是根据像素个数决定的

#b1 = tf.Variable(tf.zeros([200]))

W1 = tf.Variable(tf.truncated_normal([784,1000],stddev=0.1))#截断的正态分布

b1 = tf.Variable(tf.zeros([1000])+0.1)

L1 = tf.nn.sigmoid(tf.matmul(x,W1)+b1)

L1_drop = tf.nn.dropout(L1, keep_prob)



#W2 = tf.Variable(tf.zeros([1000,500]))#784是根据像素个数决定的

#b2 = tf.Variable(tf.zeros([500]))

W2 = tf.Variable(tf.truncated_normal([1000,1000],stddev=0.1))#截断的正态分布

b2 = tf.Variable(tf.zeros([1000])+0.1)

L2 = tf.nn.sigmoid(tf.matmul(L1_drop,W2)+b2)

L2_drop = tf.nn.dropout(L2, keep_prob)



#W3 = tf.Variable(tf.zeros([500,200]))#784是根据像素个数决定的

#b3 = tf.Variable(tf.zeros([200]))

W3 = tf.Variable(tf.truncated_normal([1000,500],stddev=0.1))#截断的正态分布

b3 = tf.Variable(tf.zeros([500])+0.1)

L3 = tf.nn.sigmoid(tf.matmul(L2_drop,W3)+b3)

L3_drop = tf.nn.dropout(L3, keep_prob)



#W4 = tf.Variable(tf.zeros([200,10]))#784是根据像素个数决定的

#b4 = tf.Variable(tf.zeros([10]))

W4 = tf.Variable(tf.truncated_normal([500,200],stddev=0.1))#截断的正态分布

b4 = tf.Variable(tf.zeros([200])+0.1)

L4 = tf.nn.sigmoid(tf.matmul(L3_drop,W4)+b4)

L4_drop = tf.nn.dropout(L4, keep_prob)



W5 = tf.Variable(tf.truncated_normal([200,10],stddev=0.1))#截断的正态分布

b5 = tf.Variable(tf.zeros([10])+0.1)



prediction = tf.nn.softmax(tf.matmul(L4_drop,W5)+b5)



#代价函数

#loss = tf.reduce_mean(tf.square(y-prediction))#二次代价hanshu

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction))

#梯度下降

train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)



#初始化变量

init = tf.global_variables_initializer()



#预测结果和标签

predict_pos = tf.argmax(prediction, 1)

label_pos = tf.argmax(y, 1)

#结果存放在bool型的列表之中

correct_prediction = tf.equal(label_pos,predict_pos)#argmax返回一维张量中最大值所在位置

#准确率

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))





with tf.Session() as sess:

    sess.run(init)

    for epoch in range(41):

        sess.run(tf.assign(lr, 0.2 * (0.95**epoch)))

        for batch in range(n_batch):

            batch_xs, batch_ys = mnist1.train.next_batch(batch_size)#按批次获得训练集

            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys, keep_prob:1.0})



        learning_rate = sess.run(lr)

        p_pos, l_pos, test_acc = sess.run([predict_pos, label_pos, accuracy], feed_dict={x:mnist1.test.images, y:mnist1.test.labels, keep_prob:1.0})#获得测试集

        test_drop_acc = sess.run(accuracy, feed_dict={x:mnist1.test.images, y:mnist1.test.labels, keep_prob:0.5})#获得训练集

        print("Iter " + str(epoch) + ",Testing Accuracy " + str(test_acc)+",Learning rate: "+str(learning_rate)+",Label is {} and number is {}".format(int(l_pos[epoch]),int(p_pos[epoch])))
