#!/usr/bin/env python
# encoding: utf-8

# @Author: ZengLei
# @Date  : 2019-01-11 11:40:43


import os
import logging
import tensorflow as tf
from . import lm_model


logger = logging.getLogger(__name__)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class Predictor(object):

    def __init__(self, session:tf.Session, model_path, w2i_path, i2w_path, expend_words_num=15, top_k_accuracy=5, top_k_recall=50):
        self.session = session
        with self.session.graph.as_default():
            self.model = lm_model.Model(False, w2i_path, i2w_path, top_k_recall=top_k_recall)
            tf.train.Saver().restore(self.session,model_path)
        self.expend_words_num = expend_words_num
        self.top_k_accuracy = top_k_accuracy

    def _encode(self,words, is_left, max_length):
        result = []
        if is_left:
            result.append(self.model.vocab.encode("<eos>"))
            for word in words[:-1]:
                result.append(self.model.vocab.encode(word))
            result.extend([0] * (max_length - len(words)))
        else:
            for word in words[1:]:
                result.append(self.model.vocab.encode(word))
            result.append(self.model.vocab.encode("<\eos>"))
            result.extend([0] * (max_length - len(words)))
        return result

    def _predict_batch_words(self, max_len, inputs_lens, words_list, ori_pos_list):
        inputs_left = [self._encode(words, True, max_len) for words in words_list]
        inputs_right = [self._encode(words, False, max_len) for words in words_list]
        feed = {
            self.model.ph_left: inputs_left,
            self.model.ph_right: inputs_right,
            self.model.ph_length: inputs_lens,
            self.model.ph_dropout: 1
        }
        y_results = self.session.run(self.model.predict, feed_dict=feed)
        error_dicts_list = []
        for i in range(len(inputs_lens)):
            ori_start = ori_pos_list[i][0]
            ori_end = ori_pos_list[i][1]
            words = words_list[i]
            error_dicts = []
            error_dict = {}
            for j in range(inputs_lens[i]):
                word_index = max(0, ori_start - self.expend_words_num)
                if not ori_start <= j + word_index  < ori_end:
                    continue
                top_list = [self.model.vocab.i2w[x] for x in y_results[1][i][j]]
                if words[j] in set(top_list[:self.top_k_accuracy]):
                    continue
                if not error_dict:
                    error_dict = {"start": j + word_index , "end": j + word_index  + 1, "word": words[j],
                                  "candidates_list": [top_list]}
                elif error_dict and error_dict["end"] == j + word_index:
                    error_dict["end"] += 1
                    error_dict["word"] += words[j]
                    error_dict["candidates_list"].append(top_list)
                else:
                    error_dicts.append(error_dict)
                    error_dict = {"start": j + word_index , "end": j + word_index  + 1, "word": words[j],
                                  "candidates_list": [top_list]}
            if error_dict:
                error_dicts.append(error_dict)
            error_dicts_list.append(error_dicts)
        return error_dicts_list

    def _put2sentence_errors_dict(self,coe_errors, errors_list, sentence_index_list, sentence_errors_dict):
        for index in range(len(errors_list)):
            sentence_index = sentence_index_list[index]
            sentence = coe_errors[sentence_index]["content"]
            errors = errors_list[index]
            if sentence_index in sentence_errors_dict:
                sentence_errors_dict[sentence_index]["error_list"].extend(errors)
            else:
                sentence_errors_dict[sentence_index] = {'index': sentence_index, 'content': sentence, "error_list": errors}

    def predict_from_cooccur(self, coe_errors):
        max_len = 0
        inputs_lens = []
        words_list = []
        ori_pos_list = []
        sentence_index_list = []
        count = 0

        sentence_errors_dict = dict()
        for sentence_index,sentence_errors in enumerate(coe_errors):
            sentence = sentence_errors["content"]
            error_list = sentence_errors["error_list"]
            if not error_list:
                sentence_errors_dict[sentence_index] = {'index': sentence_index, 'content': sentence, "error_list": []}
                continue
            for error in error_list:
                error_start = max(0, error["start"]  - self.expend_words_num)
                error_end = error["end"]  + self.expend_words_num
                error_word = sentence[error_start:error_end]
                error_word_len = len(error_word)
                max_len = max(max_len, error_word_len)

                inputs_lens.append(error_word_len)
                words_list.append(error_word)
                ori_pos_list.append((error["start"],error["end"]))
                sentence_index_list.append(sentence_index)
                if max_len*count>9000:
                    errors_list = self._predict_batch_words(max_len, inputs_lens, words_list, ori_pos_list)
                    self._put2sentence_errors_dict(coe_errors,errors_list,sentence_index_list,sentence_errors_dict)
                    max_len = 0
                    inputs_lens = []
                    words_list = []
                    ori_pos_list = []
                    sentence_index_list = []
                    count = 0
        if words_list:
            errors_list = self._predict_batch_words(max_len, inputs_lens, words_list, ori_pos_list)
            self._put2sentence_errors_dict(coe_errors,errors_list, sentence_index_list, sentence_errors_dict)
        return [sentence_errors_dict[index] for index in range(len(coe_errors))]

class Predictor_fdl(object):

    def __init__(self, session: tf.Session, model_path, expend_words_num=15, top_k_accuracy=5,
                 top_k_recall=50):
        self.session = session
        with self.session.graph.as_default():
            self.model = lm_model.LSTM_fdl(False)
            tf.train.Saver().restore(self.session, model_path)
        self.expend_words_num = expend_words_num
        self.top_k_accuracy = top_k_accuracy

    def predict(self, x, keep_prob:float):

        feed = {
            self.model.X: x,
            self.model.keep_prob: keep_prob
        }
        y_results = self.session.run(self.model.y_pre, feed_dict=feed)  * 10000
        print(y_results)
        print('-----------------')
        return y_results