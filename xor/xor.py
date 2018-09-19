# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=[4,2])
y_ = tf.placeholder(tf.float32, shape=[4,1])

#构建网络结构，两层神经网络,中间层的神经元个数为2,最后一层的神经元个数为1
with tf.variable_scope('layer1') as scope:
    weight = tf.get_variable(name='weight', shape=[2, 2], initializer=tf.random_normal_initializer)
    bias = tf.get_variable(name='bias', shape=[2], initializer=tf.constant_initializer(0.0))
    layer1_output = tf.sigmoid(tf.matmul(x, weight) + bias)
with tf.variable_scope('layer2') as scope:
    weight = tf.get_variable(name='weight', shape=[2, 1], initializer=tf.random_normal_initializer)
    bias = tf.get_variable(name='bias', shape=[1], initializer=tf.constant_initializer(0.0))
    y = tf.sigmoid(tf.matmul(layer1_output, weight) + bias)
#计算loss,这里使用的是l2 loss,即MSE
loss = tf.nn.l2_loss(y-y_)

learning_rate = 0.05
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.minimize(loss)


train_data = np.array([[0,0], [0,1], [1,0], [1,1]])
train_label = np.array([[0], [1], [1], [0]])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100000):
        loss_train, _ = sess.run([loss, train_op], feed_dict={x:train_data, y_:train_label})
        if i % 10000 == 0:
            print '{} step, loss is {}'.format(i, loss_train)
    #这里可以使用Tensorbord来查看计算图
    writer = tf.summary.FileWriter('./graph', sess.graph)
    writer.flush()
    writer.close()