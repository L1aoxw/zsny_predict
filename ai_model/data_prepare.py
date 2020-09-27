#!/usr/bin/env python
# encoding: utf-8
import collections
import re
import operator
import math
import os
import pickle
import pandas as pd
import datetime
import numpy as np
from replite import Replite, Merge_train, Merge_pred
from .lm_params import LmParams as lmp
from xpinyin import Pinyin

def pinyin_sort(lists):             #输入一个名字的列表
    pin=Pinyin()
    result=[]
    for item in lists:
        result.append((pin.get_pinyin(item),item))
    result.sort()
    for i in range(len(result)):
        result[i]=result[i][1]
    print(result)                 #输出结果
    return result


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_path = os.path.join(PROJECT_DIR, "train_data")

reg_word_split = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%，、： （）《》“”。!！？?:]+)", re.U)
split_text_chars = {'。', '!', '?', '！', '?', '\n', '\r'}
weather_history = '../train_data'
replite = Replite()
merge_train = Merge_train()

def lstm_fdl(train_data_file, train_y_file, is_addition = False):

    # replite.get_lishi()
    if is_addition:
        replite.get_lishi_addition(weather_history ,years=[2020],months=[8])
        timestr = datetime.datetime.now().strftime('%Y-%m-%d-%H')
        merge_train.get_hunan_info()
    # trian_data, train_y = merge_train.get_feature() #原来的特征处理函数可不用，现在重新编写本函数替代
    # 特征处理：
    data_csv = pd.read_csv(weather_history + '/天气历史_湖南省.csv' , sep=',', names=['hunan', 'DATA_DATETIME', 'weather', 'tp_min', 'tp_max'])
    # hunan, DATA_DATETIME, weather, tp_min, tp_max
    data_label = pd.read_csv(weather_history + '/湖南省实际用电量.csv', sep=',')
    # 删除所有存在nan值的行再合并
    data_csv.drop(data_csv[np.isnan(data_csv["tp_max"])].index, inplace=True)
    data_ = pd.merge(data_csv, data_label, how='inner', on='DATA_DATETIME')
    data_['PARAM_VALUE'] = data_['PARAM_VALUE'] / 10000
    # 训练数据保存到文件
    # data_.to_csv(weather_history + '/train_data.csv' , index=False)
    # assert len(data_csv)==len(data_label)

    years = [2015, 2016, 2017, 2018, 2019, 2020]
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    weath_all = []
    for w in list(data_[data_.columns[2]]):
        weath_all = weath_all + re.split('[转~]', w.strip())
    weath_all = list(set(list(weath_all)))
    weath_all = pinyin_sort(weath_all)
    print(weath_all)

    def cut_data(data_x,data_y):
        pass
        res_x = []
        res_y = []
        for i in range(len(data_x)):
            if any(data_x[i][0][6:11]):
                res_x.append(data_x[i])
                res_y.append(data_y[i])
        return np.array(res_x),np.array(res_y)

    def f_date(date):
        res = []
        date = date.split('/')
        m = [0 for i in months]
        m[months.index(int(date[1]))] = 1
        # 年份归一化
        y = (int(date[0]) - years[0]) / (years[-1] - years[0])

        res.append(y)
        return res + m

    def f_week(date):
        if date in ['1', '2', '3', '4', '5']:
            return [1, 0]
        else:
            return [0, 1]

    def f_weathe(data):
        # 天气编码
        res = [0 for i in weath_all]
        ws = re.split('[转~]', data.strip())
        for w in list(ws):
            res[weath_all.index(w)] = 1
        return res

    def f_tp_max(data):
        tp_max = np.array(data_csv[data_csv.columns[4]])
        res = [(float(data) - tp_max.min()) / (tp_max.max() - tp_max.min())]
        return res

    def f_tp_min(data):
        tp_min = np.array(data_csv[data_csv.columns[3]])
        res = [(float(data) - tp_min.min()) / (tp_min.max() - tp_min.min())]
        return res

    def f_label(data):
        i, j = data
        if float(i) > 10000: i = float(i) / 10
        return [float(i) / 1000, float(j) / 10000]

    feature_train = []
    label_train = []
    for i in range(len(data_)):
        line = list(data_.loc[i])
        # date feature
        if 'DATA_DATETIME' in line or '#NAME?' in line:
            pass
            continue
        f0 = f_date(line[1])
        # f1 = f_week(line[2])
        f2 = f_weathe(line[2])
        f3 = f_tp_max(line[4])
        f4 = f_tp_min(line[3])

        feature = f0 + f2 + f3 + f4
        feature_train.append(np.array(feature))
        label_train.append(np.array(line[-1]))
        # f2=f_other(line[2:])
    assert len(feature_train) == len(label_train)
    train_data = []
    train_y = []
    print('---设置样本的时间序列步长---')
    for i in range(7, len(feature_train)):
        train_data.append(feature_train[i - 7:i])
        train_y.append(label_train[i])
    print('---选取合适的月份数据---')
    train_data, train_y = cut_data(train_data, train_y)

    # 存成pickle文件
    # np.savetxt(train_data_file, train_data, delimiter=',')
    # np.savetxt(train_y_file, train_y, delimiter=',')
    pickle.dump(train_data, open(train_data_file, 'wb'))
    pickle.dump(train_y, open(train_y_file, 'wb'))
    print('---训练样本pkl文件保存---')

# def build_train_data(line_len, corpus_path, corpus_w2i_path, corpus_i2w_path, corpus_embed_path,
#                      corpus_label_path):
#     '''
#     :param line_len: 每行句子长度如果超过line_len将被分割成多句
#     :param corpus_path: 语料，要求是没有错字的文本，格式不限
#     '''
#     def _format_word(_word):
#         if _word.isdigit():
#             return "<num>"
#         if ("\u0041" <= _word <= "\u005a") or ("\u0061" <= _word <= "\u007a"):
#             return "<eng>"
#         if reg_word_split.search(_word):
#             return _word
#         else:
#             return "<unk>"
#
#     def _format_id(_id, id_max, w2i_dict):
#         return _id if int(_id) < id_max else str(w2i_dict["<unk>"])
#
#     def _confirm_save_path(path):
#         if os.path.exists(path):
#             os.remove(path)
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#
#     def _save_pkl(obj, path):
#         _confirm_save_path(path)
#         with open(path, "wb") as stream_f:
#             pickle.dump(obj, stream_f)
#
#     # word_count_dict
#     with open(corpus_path, "r", encoding="utf-8") as f_stream:
#         lines = f_stream.readlines()
#     word_count_dict = collections.defaultdict(int)
#     for idx, line in enumerate(lines):
#         line = line.strip()
#         if not line:
#             continue
#         word_count_dict["<eos>"] += 1
#         for word in line:
#             word = _format_word(word)
#             word_count_dict[word] += 1
#         word_count_dict["<\eos>"] += 1
#         word_count_dict["<pad>"] += 1
#     # dict_w2i
#     sorted_words = sorted(word_count_dict.items(), key=operator.itemgetter(1), reverse=True)
#     dict_w2i = {"<pad>": 0,"<unk>": 1}
#     word_idx = 2
#     for word,count in sorted_words:
#         if word == "<pad>" or word == "<unk>":
#             continue
#         dict_w2i[word] = word_idx
#         word_idx += 1
#     _save_pkl(dict_w2i, corpus_w2i_path)
#     # dict_i2w
#     dict_i2w = {value:key for key,value in dict_w2i.items()}
#     _save_pkl(dict_i2w, corpus_i2w_path)
#     # sentences to batch ids
#     _confirm_save_path(corpus_embed_path)
#     with open(corpus_embed_path, 'a', encoding='utf-8') as stream_embed:
#         _confirm_save_path(corpus_label_path)
#         with open(corpus_label_path, 'a', encoding='utf-8') as stream_label:
#             for idx, line in enumerate(lines):
#                 line = line.strip()
#                 if not line:
#                     continue
#                 sentences = str_util.split(line, split_text_chars)
#                 for sentence in sentences:
#                     if len(sentence) < 3:
#                         continue
#                     line_idx_list = [str(dict_w2i["<eos>"])]
#                     for word in sentence:
#                         word = _format_word(word)
#                         line_idx_list.append(str(dict_w2i[word]))
#                     line_idx_list.append(str(dict_w2i["<\eos>"]))
#                     for batch_index in range(math.ceil(len(line_idx_list) / line_len)):
#                         batch_ids = line_idx_list[batch_index * line_len: (batch_index + 1) * line_len]
#                         if len(batch_ids) <= 5:
#                             continue
#                         batch_labels = [_format_id(idx, lmp.total_class, dict_w2i) for idx in batch_ids[1: -1]]
#                         stream_embed.write(" ".join(batch_ids) + "\n")
#                         stream_label.write(" ".join(batch_labels) + "\n")