# coding:utf-8
'''
@author: liaoxingwei
@contact: 1204890081@qq.com
@time: 2020/9/22 15:45
'''
from lxml import etree
import requests
import time
import os
import datetime
import numpy as np
from urllib.parse import *
import pandas as pd
from selenium import webdriver
from bs4 import BeautifulSoup
import re
from replite import Replite
import pickle
from xpinyin import Pinyin

class Merge_pred():
    def __init__(self):
        self.replite = Replite()
        self.pred_days = 15
        self.city_dir = './data_city'
        self.hunan_dir = './data'
        self.citys = ['changsha', 'zhuzhou', 'xiangtan', 'shaoyang', 'hengyang', 'yueyang', 'changde', 'zhangjiajie', 'yiyang1',
             'chenzhou', 'yongzhou', 'huaihua', 'loudi', 'jishou']
        self.citys_w = np.array([0.23,0.07,0.07,0.05,0.1,0.08,0.07,0.02,0.05,0.06,0.05,0.06,0.07,0.03])
    def get_hunan_info(self,predict_file,timestr):
        pass
        citys_data = self.city_dir+'/天气预报15天_各市州_%s.csv'%(timestr)
        self.replite.get_yubao(self.city_dir,timestr)
        lines = pd.read_csv(citys_data,header=None,sep=',')
        res=[]
        for idx in range(len(lines)//14):
            hunan_day = lines[idx::(self.pred_days-1)]
            # header:[地区，日期，星期，天气，最低温，最高温]
            hunan_city = 'hunan'
            hunan_date = str(hunan_day.iat[0,1])
            hunan_week = str(hunan_day.iat[0,2])
            hunan_weathe = str(hunan_day[3].value_counts().index[0])
            hunan_min = str(round(sum(hunan_day[4]._values * self.citys_w), 2))
            hunan_max = str(round(sum(hunan_day[5]._values * self.citys_w), 2))
            res.append(','.join([hunan_city,hunan_date,hunan_weathe,hunan_min,hunan_max]))
        open(predict_file, 'w', encoding='utf-8').write('\n'.join(res))
        return predict_file
    def get_feature(self,csv_file='./天气预报15天_湖南省.csv'):
        data_csv = pd.read_csv(csv_file, header=None,sep=',')
        train_data = pd.read_csv('./train_data.csv', sep=',')
        years = [2015, 2016, 2017, 2018, 2019, 2020]
        months = [1,2,3,4,5,6,7,8,9,10,11,12]
        weath_all = []
        for w in list(train_data[train_data.columns[2]]):
            weath_all = weath_all+re.split('[转~]',w.strip())
        weath_all = list(set(list(weath_all)))
        weath_all = pinyin_sort(weath_all)

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
                return [1,0]
            else:
                return [0,1]

        def f_weathe(data):
            # 天气编码
            res = [0 for i in weath_all]
            ws = re.split('[转到~]',data.strip())
            for w in list(ws):
                if w=='雨': w='小雨'
                res[weath_all.index(w)] = 1
            return res

        def f_tp_max(data):
            tp_max = np.array(train_data[train_data.columns[4]])
            res=[(float(data) - tp_max.min()) / (tp_max.max()-tp_max.min())]
            return res

        def f_tp_min(data):
            tp_min = np.array(train_data[train_data.columns[3]])
            res = [(float(data) - tp_min.min()) / (tp_min.max()-tp_min.min())]
            return res

        def f_label(data):
            i, j = data
            if float(i) > 10000: i = float(i) / 10
            return [float(i) / 1000, float(j) / 10000]

        feature_result = []
        for i in range(len(data_csv)):
            line = list(data_csv.loc[i])
            # date feature
            # fy = f_label(line[-2:])

            f0 = f_date(line[1])
            # f1 = f_week(line[2])
            f2 = f_weathe(line[2])
            f3 = f_tp_max(line[4])
            f4 = f_tp_min(line[3])

            feature = f0 + f2 + f3 + f4
            feature_result.append(np.array(feature))
            # f2=f_other(line[2:])
        feature_result = np.array(feature_result)
        pass
        return feature_result

class Merge_train():
    def __init__(self):
        self.pred_days = 15
        self.citys_w = np.array([0.23,0.07,0.07,0.05,0.1,0.08,0.07,0.02,0.05,0.06,0.05,0.06,0.07,0.03])
    def get_hunan_info(self):
        citys = ['changsha', 'zhuzhou', 'xiangtan', 'shaoyang', 'hengyang', 'yueyang', 'changde', 'zhangjiajie',
                      'yiyang',
                      'chenzhou', 'yongzhou', 'huaihua', 'loudi', 'jishou']
        if not os.path.exists('201501-202008各市州气候.csv'):
            txt_file = './天气历史_%s.csv' % (citys[0])
            data = pd.read_csv(txt_file, header=None, sep=',')
            for i in range(1, len(citys)):
                txt_file =  './天气历史_%s.csv' % (citys[i])
                temp = pd.read_csv(txt_file, header=None, sep=',')
                data = pd.merge(data, temp, how='left', on=[data.columns[1], temp.columns[1]])
            # labels = pd.read_csv('label_2019.csv',header=None)
            # data=pd.merge(data, labels, how='inner', on=[data.columns[0], labels.columns[0]])
            data.to_csv('201501-202008各市州气候.csv', index=False, header=False)
        data = pd.read_csv('201501-202008各市州气候.csv', header=None, sep=',')
        res = []
        for i in range(len(data)):
            #'changsha', '2015/1/1', '多云', 4, 12, 'zhuzhou', '多云', 4, 12, 'xiangtan', '多云', 5, 12, 'shaoyang', '阴', 11.0, 16.0,...
            line = list(data.iloc[i])
            list_weathe = line[2::4]
            list_min = line[3::4]
            list_max = line[4::4]
            from collections import Counter
            collection_words = Counter(list_weathe)
            hunan_city = 'hunan'
            hunan_date = line[1]
            hunan_weathe = collection_words.most_common(1)[0][0]
            hunan_min = str(round(sum(list_min * self.citys_w), 2))
            hunan_max = str(round(sum(list_max * self.citys_w), 2))
            res.append(','.join([hunan_city, hunan_date, hunan_weathe, hunan_min, hunan_max]))
        open('./天气历史_湖南省.csv', 'w', encoding='utf-8').write('\n'.join(res))
        print('---')

    def get_feature(self,csv_file='./天气历史_湖南省.csv',label_file='./201611-202008湖南省统调最大负荷.csv'):
        data_csv = pd.read_csv(csv_file, sep=',')
        data_label = pd.read_csv(label_file,sep=',')
        # 删除所有存在nan值的行再合并
        data_csv.drop(data_csv[np.isnan(data_csv["tp_max"])].index, inplace=True)
        data_ = pd.merge(data_csv, data_label, how='inner', on='DATA_DATETIME')
        # 训练数据保存到文件
        data_.to_csv('./train_data.csv', index=False)
        # assert len(data_csv)==len(data_label)
        years = [2015, 2016, 2017, 2018, 2019, 2020]
        months = [1,2,3,4,5,6,7,8,9,10,11,12]

        def pinyin_sort(lists):  # 输入一个名字的列表
            pin = Pinyin()
            result = []
            for item in lists:
                result.append((pin.get_pinyin(item), item))
            result.sort()
            for i in range(len(result)):
                result[i] = result[i][1]
            print(result)  # 输出结果
            return result

        weath_all = []
        for w in list(data_[data_.columns[2]]):
            weath_all = weath_all+re.split('[转~]',w.strip())
        weath_all = list(set(list(weath_all)))
        weath_all = pinyin_sort(weath_all)
        print(weath_all)

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
                return [1,0]
            else:
                return [0,1]

        def f_weathe(data):
            # 天气编码
            res = [0 for i in weath_all]
            ws = re.split('[转~]',data.strip())
            for w in list(ws):
                res[weath_all.index(w)] = 1
            return res

        def f_tp_max(data):
            tp_max = np.array(data_csv[data_csv.columns[4]])
            res=[(float(data) - tp_max.min()) / (tp_max.max()-tp_max.min())]
            return res

        def f_tp_min(data):
            tp_min = np.array(data_csv[data_csv.columns[3]])
            res = [(float(data) - tp_min.min()) / (tp_min.max()-tp_min.min())]
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
        train_data = np.array(feature_train)
        train_y = np.array(label_train)
        assert len(train_data)==len(train_y)
        return train_data,train_y

if __name__ == '__main__':

    replite = Replite()
    # replite.get_lishi()
    # replite.get_lishi_addition(years=[2020],months=[8,9])
    merge_train = Merge_train()
    merge_pred = Merge_pred()
    timestr = datetime.datetime.now().strftime('%Y-%m-%d-%H')
    merge_pred.get_hunan_info(timestr)
    # replite.get_lishi()
    if not os.path.exists('./天气历史_湖南省.csv'):
        merge_train.get_hunan_info()
    if not os.path.exists('./天气预报15天_湖南省.csv'):
        merge_pred.get_hunan_info()
    trian_data,train_y = merge_train.get_feature()
    predict_data = merge_pred.get_feature()
    print('---')