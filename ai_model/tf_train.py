#!/usr/bin/env python
# encoding: utf-8


import os
import logging
import tensorflow as tf
import numpy as np
import replite
import pickle
import matplotlib.pyplot as plt

from ai_model import data_prepare
from ai_model import lm_model
from ai_model.lm_params import LmParams as lmp


logger = logging.getLogger(__name__)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def get_batch(faces,label,batch_size):#训练时用
    array=np.arange(label.shape[0])
    np.random.shuffle(array)
    idx=array[:batch_size]
    return faces[idx], label[idx] # 生成每一个batch

def get_vaild_batch(faces,label,batch_size):#训练时用
    array=np.arange(label.shape[0])
    idx=array[:batch_size]
    return faces[idx], label[idx] # 生成每一个batch

def LSTM_fdl_train():

    model_path = os.path.join(PROJECT_DIR, "data.out.train/lstm_fdl/lstm_fdl.weights")
    data_path = os.path.join(PROJECT_DIR, "data.in.train/lstm_fdl")

    train_data_file = data_path + '/train_data.pkl'
    train_y_file = data_path + '/train_y.pkl'


    #预处理训练数据
    if not os.path.exists(train_data_file) or not os.path.exists(train_y_file):
        logger.info('数据预处理')
        data_prepare.lstm_fdl(train_data_file, train_y_file)

    #读取预处理完成的pickle训练样本
    def train_and_test(data,label, ratio=0.1):
        data_train = []
        data_test = []
        label_train = []
        label_test = []
        cnt = 0
        for i in range(data.shape[0]):
            if (data[i][0][0] == 1) and data[i][6][8]:
                if cnt >= 0:
                    data_test.append(data[i])
                    label_test.append(label[i])
                else:
                    cnt += 1
                    data_train.append(data[i])
                    label_train.append(label[i])
                    data_train.append(data[i])  # 扩充
                    label_train.append(label[i])
            else:
                data_train.append(data[i])
                label_train.append(label[i])

        data_train = np.array(data_train)
        label_train = np.array(label_train)
        data_test = np.array(data_test)
        label_test = np.array(label_test)
        return data_train, data_test, label_train, label_test
    logger.info('训练数据读取')
    train_data = pickle.load(open(train_data_file, 'rb'))
    train_y = pickle.load(open(train_y_file, 'rb'))
    data_train, data_test, label_train, label_test = train_and_test(train_data, train_y)

    #训练
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(minval=-lmp.init_weight, maxval=lmp.init_weight,
                                                    seed=lmp.tf_random_seed)
        tf.get_variable_scope().set_initializer(initializer)

        model = lm_model.LSTM_fdl(is_train=True)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        saver = tf.train.Saver()
        with tf.Session(config=tf_config) as session:
            logger.info("start")
            session.run(tf.global_variables_initializer())
            feed = {model.keep_prob: lmp.dropout_rate}
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            max_iter = 100000
            eoch_i = np.zeros((max_iter, 1))
            eoch_loss = np.zeros((max_iter, 1))
            for i in range(max_iter):
                batch_x, batch_y = get_batch(data_train, label_train, lmp.batch_size)  # 随机取batch_size个样本和标签
                vaild_x, vaild_y = get_vaild_batch(data_test, label_test, lmp.batch_size)
                feed_dict = {model.X: batch_x, model.y: batch_y, model.keep_prob: lmp.dropout_rate}
                vaild_dict = {model.X: vaild_x, model.y: vaild_y, model.keep_prob: 1}
                eoch_loss[i] = session.run(model.loss, feed_dict)
                if i % 200 == 0:
                    train_loss = session.run(model.loss, feed_dict = feed_dict)
                    vaild_loss = session.run(model.loss, feed_dict = vaild_dict)
                    print('===train loss:', train_loss, 'vaild loss:', vaild_loss)
                if i % 2000 == 0:
                    vaild_pred = session.run(model.y_pre, feed_dict=vaild_dict)
                    print(vaild_y)
                    print(vaild_pred)
                    print(np.mean(abs(vaild_y - vaild_pred) / vaild_y))
                    plt.plot(vaild_y, 'b', vaild_pred, 'r')
                    plt.show()
                    epoch_dir = data_path + '/epoch=' + str(i) + '/'
                    if os.path.exists(epoch_dir) == 0:
                        os.makedirs(epoch_dir)
                    saver.save(session, epoch_dir + 'model.ckpt')
                # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                session.run(model.train_op, feed_dict=feed_dict)


def train():
    #输入：任意没有错字的文本
    corpus_path = os.path.join(PROJECT_DIR,"data.in.train/corpus.txt")
    #输出文件：
    corpus_w2i_path = os.path.join(PROJECT_DIR,"data.out.train.prepare/corpus_w2i.pkl")
    corpus_i2w_path = os.path.join(PROJECT_DIR,"data.out.train.prepare/corpus_i2w.pkl")
    corpus_embed_path = os.path.join(PROJECT_DIR,"data.out.train.prepare/corpus_embed.txt")
    corpus_label_path = os.path.join(PROJECT_DIR,"data.out.train.prepare/corpus_label.txt")

    model_path = os.path.join(PROJECT_DIR, "data.out.train/lstm.weights")
    #预处理训练数据
    if not os.path.exists(corpus_w2i_path) \
            or not os.path.exists(corpus_i2w_path) \
            or not os.path.exists(corpus_embed_path) \
            or not os.path.exists(corpus_label_path):
        data_prepare.build_train_data(30, corpus_path, corpus_w2i_path, corpus_i2w_path, corpus_embed_path,
                     corpus_label_path)
    #训练
    with tf.Graph().as_default():
        initializer = tf.random_uniform_initializer(minval=-lmp.init_weight, maxval=lmp.init_weight,
                                                    seed=lmp.tf_random_seed)
        tf.get_variable_scope().set_initializer(initializer)

        model = lm_model.Model(True, corpus_w2i_path, corpus_i2w_path, embed_path=corpus_embed_path, label_path=corpus_label_path)

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        saver = tf.train.Saver()
        with tf.Session(config=tf_config) as session:
            logger.info("start")
            session.run(tf.global_variables_initializer())
            feed = {model.ph_dropout: lmp.dropout_rate}
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            for epoch in range(lmp.max_epoch):
                try:
                    session.run(model.iterator.initializer)
                    while True:
                        session.run([model.outputs, model.loss, model.train_step], feed_dict=feed)
                        logger.info("running")
                except Exception as e:
                    logger.exception("")
                saver.save(session, model_path)
                logger.info("finish")

if __name__ == "__main__":
    # train()
    LSTM_fdl_train()