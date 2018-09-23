# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data


def load_data():
    mnist = input_data.read_data_sets('./data', one_hot=True)
    return mnist

def KNN(mnist):
    train_x, train_y = mnist.train.next_batch(5000)#从训练集中取出5000个作为参考样本
    test_x, test_y = mnist.train.next_batch(200)  #从训练集中取出200个用于测试
    
    x = tf.placeholder(tf.float32, [None, 784], name='train-x')
    x_ = tf.placeholder(tf.float32, [784], name='test-x')
    #使用L2距离作为度量
    distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.add(x, tf.negative(x_)), 2), reduction_indices=1))
    #获得距离最近的训练样本索引
    min_idx = tf.argmin(distance, 0)
    
    right = 0 #用于统计正确分类的个数
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(len(test_x)):
            nn_idx = sess.run([min_idx], feed_dict={x:train_x, x_:test_x[i, :]})#对每一个测试样本算出其距离最近（k=1）的样本索引并将其类别作为该测试样本的类别
            predict_class = np.argmax(train_y[nn_idx])
            true_class = np.argmax(test_y[i])
            print 'test {}, predict class is {}, true class is {}'.format(i, predict_class, true_class)
            if predict_class == true_class:
                right += 1
        accuracy = right / float(len(test_x))#计算准确率
        print "accuracy is ", accuracy
        
if __name__ == '__main__':
    mnist = load_data()
    KNN(mnist)