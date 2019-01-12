# coding:utf-8
# auther:xavier, 20190112
import numpy as np
import time
import random
import tensorflow as tf
import matplotlib.pyplot as plt


def dataSet(size):
    data = []
    for i in range(size):
        x = i / 1000.0
        data.append((x, 2 * x ** 2))
        data.append((x, x ** 2))
    return data


def generetor(x, y, name="gen"):
    with tf.variable_scope(name):
        w1 = tf.get_variable(shape=[1, 10], name="w1")
        b1 = tf.get_variable(shape=[10], name="b1")
        pred = tf.nn.sigmoid(tf.matmul(x, w1) + b1)
        w2 = tf.get_variable(shape=[10, 10], name="w2")
        b2 = tf.get_variable(shape=[10], name="b2")
        pred = tf.matmul(pred, w2) + b2
        w3 = tf.get_variable(shape=[10, 1], name="w3")
        b3 = tf.get_variable(shape=[1], name="b3")
        pred = tf.matmul(pred, w3) + b3
        loss = tf.sqrt(tf.reduce_mean((y - pred) ** 2))
        return loss, pred


def discriminator(input, label, name="disc"):
    with tf.variable_scope(name):
        w1 = tf.get_variable(shape=[2, 10], name="w1")
        b1 = tf.get_variable(shape=[10], name="b1")
        hidden = tf.nn.sigmoid(tf.matmul(input, w1) + b1)
        w2 = tf.get_variable(shape=[10, 10], name="w2")
        b2 = tf.get_variable(shape=[10], name="b2")
        hidden = tf.nn.sigmoid(tf.matmul(hidden, w2) + b2)
        w3 = tf.get_variable(shape=[10, 2], name="w3")
        b3 = tf.get_variable(shape=[2], name="b3")
        pred = tf.nn.softmax(tf.matmul(hidden, w3) + b3)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label))
    return loss, pred


def train():
    tf.set_random_seed(123)
    num_step = 10000
    data = dataSet(10000)
    with tf.variable_scope("adv", reuse=tf.AUTO_REUSE):
        x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="x")
        y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="y")
        gloss, gpred = generetor(x, y, name="gen")

        disc_input = tf.placeholder(dtype=tf.float32, shape=[None, 2], name="disc_input")
        disc_label = tf.placeholder(dtype=tf.int32, shape=[None], name="label")
        dloss, dpred = discriminator(disc_input, disc_label)

        gparams, dparams = [], []
        for param in tf.global_variables():
            print(param.name)
            if param.name.startswith("adv/disc"):
                dparams.append(param)
            if param.name.startswith("adv/gen"):
                gparams.append(param)

        dopt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(dloss, var_list=dparams)
        gopt = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(gloss, var_list=gparams)

        adv_data_0 = tf.concat([x, gpred], axis=1)
        adv_data_1 = tf.concat([x, y], axis=1)
        adv_data = tf.concat([ adv_data_1, adv_data_0], axis=0)
        adv_dloss, _ = discriminator(adv_data, disc_label)
        adv_gloss, _ = discriminator(adv_data_0, disc_label)

        gen_opt = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(adv_gloss, var_list=gparams)
        dis_opt = tf.train.AdamOptimizer(learning_rate=0.001).minimize(adv_dloss, var_list=dparams)

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())
        print("pretrain generator ...")
        for i in range(num_step):
            batch_data = random.sample(data, 64)
            batch_x = [[ele[0]] for ele in batch_data]
            batch_y = [[ele[1]] for ele in batch_data]
            feed_dict = {x.name: batch_x, y.name: batch_y}
            gcost, _ = sess.run([gloss, gopt], feed_dict=feed_dict)

            if (i % 100 == 0):
                print("pretrain generator loss:g1loss=%f" % gcost)

        print("generate data for discriminator ...")
        input_x = []
        for ele in data:
            input_x.append(ele[0])
        input_x = list(set(input_x))
        input_x = [[ele] for ele in input_x]
        pred_y = sess.run(gpred, feed_dict={x.name: input_x})
        data_x = [ele[0] for ele in data]
        data_y = [ele[1] for ele in data]
        input_x = [ele[0] for ele in input_x]
        pred_y = [ele[0] for ele in pred_y]
        plt.scatter(data_x, data_y, c='b')
        plt.scatter(input_x, list(pred_y), c='g')
        gen_data = list(zip(input_x, list(pred_y)))

        print("pretrain discriminator ...")
        for i in range(num_step*2):
            batch_x = random.sample(data, 32) + random.sample(gen_data, 32)
            batch_label = 32 * [0] + 32 * [1]
            feed_dict = {disc_input.name: batch_x, disc_label.name: batch_label}
            dcost, _ = sess.run([dloss, dopt], feed_dict=feed_dict)
            if (i % 100 == 0):
                print("pretrain discriminator loss:", dcost)

        print("adversarial training ...")
        for i in range(num_step*3):
            batch_data = random.sample(data, 64)
            batch_x = [[ele[0]] for ele in batch_data]
            batch_y = [[ele[1]] for ele in batch_data]
            feed_dict = {x.name: batch_x, y.name: batch_y, disc_label.name: 64 * [0]}#positive=0, maximize the error of discriminator
            gcost, _ = sess.run([gloss, gen_opt], feed_dict=feed_dict)

            if (i % 5 == 0):
                feed_dict = {x.name: batch_x, y.name: batch_y, disc_label.name: 64 * [0] + 64 * [1]}
                dcost, _ = sess.run([adv_dloss, dis_opt], feed_dict=feed_dict)
            if (i % 100 == 0):
                print("adversarial training loss:gloss=%f,dloss=%f" % (gcost, dcost))

        print("generate data after adv ...")
        input_x = []
        for ele in data:
            input_x.append(ele[0])
        input_x = list(set(input_x))
        input_x = [[ele] for ele in input_x]
        pred_y = sess.run(gpred, feed_dict={x.name: input_x})
        input_x = [ele[0] for ele in input_x]
        pred_y = [ele[0] for ele in pred_y]
        plt.scatter(input_x, list(pred_y), c='r')
        plt.savefig("fig.png")


if __name__ == "__main__":
    train()
