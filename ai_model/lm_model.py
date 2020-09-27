#!/usr/bin/env python
# encoding: utf-8

import pickle
import tensorflow as tf
from tensorflow.contrib import rnn

from .lm_params import LmParams as lmp

def _create_iterator(batch_size, embed_path, label_path):
    def __batch_func(x):
        padded_shape = tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([None]), tf.TensorShape([])
        return x.padded_batch(batch_size, padded_shapes=(padded_shape), padding_values=(0, 0, 0, 0))

    def __convert2int(x, y):
        int_x = tf.to_int32(tf.string_to_number((tf.string_split([x]).values)))
        int_y = tf.to_int32(tf.string_to_number((tf.string_split([y]).values)))
        return int_x, int_y

    def __reshape(x, y):
        part_a = x[:-2]
        part_b = x[2:]
        return part_a, part_b, y, tf.size(part_a)

    data_txt = tf.data.Dataset.from_tensor_slices(embed_path)
    data_txt = data_txt.flat_map(lambda filename: (tf.data.TextLineDataset(filename)))
    data_label = tf.data.Dataset.from_tensor_slices(label_path)
    data_label = data_label.flat_map(lambda filename: (tf.data.TextLineDataset(filename)))
    tf_data = tf.data.Dataset.zip((data_txt, data_label))
    tf_data = tf_data.map(lambda x, y: __convert2int(x, y))
    tf_data = tf_data.map(lambda x, y: __reshape(x, y))

    batch_data = __batch_func(tf_data.shuffle(buffer_size=300 * batch_size))
    iterator = batch_data.make_initializable_iterator()
    return iterator
class Vocab(object):
    def __init__(self,w2i_file,i2w_file):
        with open(w2i_file,"rb") as f_stream:
            self.w2i = pickle.load(f_stream)
        with open(i2w_file,"rb") as f_stream:
            self.i2w = pickle.load(f_stream)

    def encode(self,word):
        if word.isdigit():
            word =  "<num>"
        if ("\u0041" <= word <= "\u005a") or ("\u0061" <= word <= "\u007a"):
            word =  "<eng>"
        if word not in self.w2i:
            word = "<unk>"
        return self.w2i[word]

    def decode(self,index):
        return self.i2w[index]

    def __len__(self):
        return len(self.w2i)
class Model(object):

    def __init__(self, is_train, w2i_path, i2w_path,top_k_recall=200,embed_path=None,label_path=None):
        self.vocab = Vocab(w2i_path, i2w_path)

        if is_train:
            iterator = _create_iterator(lmp.batch_size, [embed_path], [label_path])
            ph_left, ph_right, ph_labels, ph_length = iterator.get_next()
            ph_dropout = tf.placeholder(tf.float32, None)
        else:
            iterator = None
            ph_left = tf.placeholder(tf.int32,shape=(None,None))
            ph_right = tf.placeholder(tf.int32,shape=(None,None))
            ph_labels = None
            ph_length = tf.placeholder(tf.int32,(None))
            ph_dropout = tf.placeholder(tf.float32,None)

        with tf.variable_scope("embedding"):
            embedding = tf.get_variable("LM", [len(self.vocab), lmp.embed_size])
            inputs_left = tf.nn.embedding_lookup(embedding, ph_left)
            inputs_right = tf.nn.embedding_lookup(embedding, ph_right)
            inputs_right = tf.reverse_sequence(inputs_right,ph_length, batch_dim=0, seq_dim=1)
        with tf.variable_scope("lstm_left"):
            cell_left = tf.contrib.rnn.LSTMBlockCell(lmp.hidden_size)
            cell_drop_left = tf.contrib.rnn.DropoutWrapper(cell_left, input_keep_prob=ph_dropout)
            rnn_left, state_left = tf.nn.dynamic_rnn(cell_drop_left, inputs_left, ph_length, dtype=tf.float32)
        with tf.variable_scope("lstm_right"):
            cell_right = tf.contrib.rnn.LSTMBlockCell(lmp.hidden_size)
            cell_drop_right = tf.contrib.rnn.DropoutWrapper(cell_right, input_keep_prob=ph_dropout)
            rnn_right, state_right = tf.nn.dynamic_rnn(cell_drop_right, inputs_right, dtype=tf.float32)
        tensor_concat = tf.concat([rnn_left, tf.reverse_sequence(rnn_right,ph_length, batch_dim=0, seq_dim=1)], -1)
        outputs = tf.layers.dense(tensor_concat, lmp.total_class)

        if is_train:
            with tf.name_scope("losses"):
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ph_labels, logits=outputs)
                max_time = ph_labels.shape[1].value or tf.shape(ph_labels)[1]
                target_weights = tf.sequence_mask(ph_length, max_time, dtype=outputs.dtype)
                loss = tf.reduce_sum(cross_entropy * target_weights / tf.to_float(lmp.batch_size))
            with tf.variable_scope("training", reuse=None):
                optimizer = tf.train.AdamOptimizer(lmp.learn_rate)
                params = tf.trainable_variables()
                gradients = tf.gradients(loss, params)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, lmp.max_gradient_norm)
                train_step = optimizer.apply_gradients(zip(clipped_gradients, params))
            self.ph_dropout = ph_dropout
            self.iterator = iterator
            self.outputs = outputs
            self.loss = loss
            self.train_step =train_step
        else:
            self.predict = tf.nn.top_k(outputs,k=top_k_recall)
            self.ph_left = ph_left
            self.ph_right = ph_right
            self.ph_length = ph_length
            self.ph_dropout = ph_dropout

class LSTM_fdl(object):
    def __init__(self, is_train,embed_path=None,label_path=None):

        if is_train:
            X = tf.placeholder(tf.float32, [None, 7, 29])
            y = tf.placeholder(tf.float32, [None])
            keep_prob = tf.placeholder(tf.float32, [])
        else:
            X = tf.placeholder(tf.float32, [None, 7, 29])
            keep_prob = tf.placeholder(tf.float32, [])

        with tf.variable_scope("lstm"):
            def lstm_cell():
                cell = rnn.LSTMCell(lmp.hidden_size, reuse=tf.get_variable_scope().reuse)
                return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

            mlstm_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(lmp.layer_num)], state_is_tuple=True)
            init_state = mlstm_cell.zero_state(7, dtype=tf.float32)

            # **步骤6：方法一，调用 dynamic_rnn() 来让我们构建好的网络运行起来
            # ** 当 time_major==False 时， outputs.shape = [batch_size, timestep_size, hidden_size]
            # ** 所以，可以取 h_state = outputs[:, -1, :] 作为最后输出
            # ** state.shape = [layer_num, 2, batch_size, hidden_size],
            # ** 或者，可以取 h_state = state[-1][1] 作为最后输出
            # ** 最后输出维度是 [batch_size, hidden_size]
            # outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=X, initial_state=init_state, time_major=False)
            # h_state = outputs[:, -1, :]  # 或者 h_state = state[-1][1]

            # *************** 为了更好的理解 LSTM 工作原理，我们把上面 步骤6 中的函数自己来实现 ***************
            # 通过查看文档你会发现， RNNCell 都提供了一个 __call__()函数（见最后附），我们可以用它来展开实现LSTM按时间步迭代。
            # **步骤6：方法二，按时间步展开计算
            outputs = list()
            state = init_state
            with tf.variable_scope('RNN'):
                for timestep in range(lmp.timestep_size):
                    if timestep > 0:
                        tf.get_variable_scope().reuse_variables()
                    # 这里的state保存了每一层 LSTM 的状态
                    (cell_output, state) = mlstm_cell(X[:, timestep, :], state)
                    outputs.append(cell_output)
            h_state = outputs[-1]

        # 上面 LSTM 部分的输出会是一个 [hidden_size] 的tensor，我们要分类的话，还需要接一个 softmax 层
        # 首先定义 softmax 的连接权重矩阵和偏置
        # out_W = tf.placeholder(tf.float32, [hidden_size, class_num], name='out_Weights')
        # out_bias = tf.placeholder(tf.float32, [class_num], name='out_bias')
        # 开始训练和测试
        with tf.variable_scope("last_predict"):
            W = tf.Variable(tf.truncated_normal([lmp.hidden_size, lmp.class_num], stddev=0.1), dtype=tf.float32)
            bias = tf.Variable(tf.constant(0.1, shape=[lmp.class_num]), dtype=tf.float32)
            y_pre = tf.matmul(h_state, W) + bias

        if is_train:
            with tf.name_scope("losses"):
                # loss = tf.reduce_mean(tf.square(tf.reshape(y_pre, [-1]) - tf.reshape(y, [-1])))
                print(tf.math.abs(tf.reshape(y_pre, [-1]) - tf.reshape(y, [-1]))/tf.reshape(y, [-1]))
                loss = tf.reduce_mean(tf.math.abs(tf.reshape(y_pre, [-1]) - tf.reshape(y, [-1]))/tf.reshape(y, [-1]))
                # train_op = tf.train.AdamOptimizer(lmp.learn_rate).minimize(cross_entropy)
            with tf.variable_scope("training", reuse=None):
                optimizer = tf.train.AdamOptimizer(lmp.learn_rate)
                params = tf.trainable_variables()
                gradients = tf.gradients(loss, params)
                clipped_gradients, _ = tf.clip_by_global_norm(gradients, lmp.max_gradient_norm)
                train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

            self.X = X
            self.y = y
            self.keep_prob = keep_prob
            self.y_pre = y_pre
            self.loss = loss
            self.train_op =train_op
        else:
            self.X = X
            self.keep_prob = keep_prob
            self.y_pre = y_pre
