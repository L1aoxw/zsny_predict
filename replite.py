# -*- conding:utf-8 -*-
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
import pickle
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

class Replite():
    def __init__(self):
        self.pred_days = 15
        self.citys_w = np.array([0.23,0.07,0.07,0.05,0.1,0.08,0.07,0.02,0.05,0.06,0.05,0.06,0.07,0.03])
    def _get_yubao(self,city):
        url = 'http://www.tianqi.com/%s/%d.html' % (city,self.pred_days)
        # 不同天气网站的城市编码不一样
        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.98 Safari/537.36 LBBROWSER'
        }
        r=''
        # 解决requests异常requests.exceptions.ConnectionError: HTTPSConnectionPool Max retries exceeded
        while r=='':
            try:
                r = requests.get(url,headers=headers)
            except Exception as e:
                time.sleep(5)
                print('is zzZZZZ...',url,e)
                continue
        r.encoding = 'utf-8'  #
        soup = BeautifulSoup(r.text, 'lxml')
        date = soup.select('body > div.w1100 > div.inleft > ul.weaul > li > a > div.weaul_z ')
        res = []
        for i in range(len(date)):
            str = date[i].get_text()
            p = re.compile(u'[\u4e00-\u9fa5]+')
            q = re.compile(u'[0-9-]+')
            res_date = (datetime.datetime.now()+datetime.timedelta(days=i)).strftime("%Y/%m/%d")
            res_week = (datetime.datetime.now() + datetime.timedelta(days=i)).strftime("%w")
            res_weathe = ','.join(re.findall(p, str))
            res_temp = ','.join(re.findall(q, str))
            res.append(','.join([city,res_date,res_week,res_weathe,res_temp]))
        return '\n'.join(res)

    def _get_yubao_2(self,city):
        # WWW.weather.com.cn 爬取天气预报，政务云开放的网站
        city2id = {'changsha':'101250101', 'zhuzhou':'101250301', 'xiangtan':'101250201', 'shaoyang':'101250901',
                 'hengyang':'101250401', 'yueyang':'101251001', 'changde':'101250601', 'zhangjiajie':'101251101',
                 'yiyang1':'101250701',
                 'chenzhou':'101250501', 'yongzhou':'101251401', 'huaihua':'101251201', 'loudi':'101250801', 'jishou':'101251501'}
        url_7d = 'http://www.weather.com.cn/weather/%s.shtml' % (city2id[city])
        url_8_15d = 'http://www.weather.com.cn/weather%dd/%s.shtml' % (self.pred_days, city2id[city])
        # 不同天气网站的城市编码不一样
        def request_url(url):
            headers = {
                'user-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.98 Safari/537.36 LBBROWSER'
            }
            r=''
            # 解决requests异常requests.exceptions.ConnectionError: HTTPSConnectionPool Max retries exceeded
            while r=='':
                try:
                    r = requests.get(url,headers=headers)
                except Exception as e:
                    time.sleep(5)
                    print('is zzZZZZ...',url,e)
                    continue
            return r
        # r.encoding = 'utf-8'  #
        # soup = BeautifulSoup(r.text, 'lxml')
        # date = soup.select('body > div.w1100 > div.inleft > ul.weaul > li > a > div.weaul_z ')
        text_7d = request_url(url_7d).content.decode("utf-8", "ignore")
        html_7d = etree.HTML(text_7d)
        text_15d = request_url(url_8_15d).content.decode("utf-8", "ignore")
        html_15d = etree.HTML(text_15d)
        # date = html.xpath("//table[@class='b']//tr/td/a/text()")
        # html_7d.xpath("//*[@id='7d']/ul/li[1]/p[1]/text()")    #天气 晴转多云
        # html_7d.xpath("//*[@id='7d']/ul/li[1]/p[2]/span/text()")   #最高温 36
        # html_7d.xpath("//*[@id='7d']/ul/li[1]/p[2]/i/text()")  #最低温 27℃
        # html_15d.xpath("//*[@id='15d']/ul/li[1]/span[2]/text()")    #天气
        # html_15d.xpath("//*[@id='15d']/ul/li[1]/span[3]/em/text()") #最高温 36
        # html_15d.xpath("//*[@id='15d']/ul/li[1]/span[3]/text()")    #最低温 /27℃

        res = []
        for i in range(1,self.pred_days):
            p = re.compile(u'[\u4e00-\u9fa5]+')
            q = re.compile(u'[0-9-]+')
            res_date = (datetime.datetime.now() + datetime.timedelta(days=i)).strftime("%Y/%m/%d")
            res_week = (datetime.datetime.now() + datetime.timedelta(days=i)).strftime("%w")
            if i<7:
                res_weathe = str(html_7d.xpath("//*[@id='7d']/ul/li["+str(i+1)+"]/p[1]/text()")[0])    #天气 晴转多云
                res_temp_max = str(html_7d.xpath("//*[@id='7d']/ul/li["+str(i+1)+"]/p[2]/span/text()")[0])  #最高温 36
                res_temp_mim = str(html_7d.xpath("//*[@id='7d']/ul/li["+str(i+1)+"]/p[2]/i/text()")[0])  #最低温 27℃
                res_temp_max = res_temp_max.replace('℃', '').replace('/', '')
                res_temp_mim = res_temp_mim.replace('℃','').replace('/','')
                pass
            else:
                i -= 7
                res_weathe = str(html_15d.xpath("//*[@id='15d']/ul/li["+str(i+1)+"]/span[2]/text()")[0])    #天气
                res_temp_max = str(html_15d.xpath("//*[@id='15d']/ul/li["+str(i+1)+"]/span[3]/em/text()")[0]) #最高温 36
                res_temp_mim = str(html_15d.xpath("//*[@id='15d']/ul/li["+str(i+1)+"]/span[3]/text()")[0])    #最低温 /27℃
                res_temp_max = res_temp_max.replace('℃', '').replace('/', '')
                res_temp_mim = res_temp_mim.replace('℃', '').replace('/', '')
                pass
            res.append(','.join([city, res_date, res_weathe, res_temp_mim, res_temp_max]))
        return '\n'.join(res)


    def _get_lishi(self,city:str,year:int,month:int):
        if city == 'yiyang1': city = 'yiyang'
        url = 'http://www.tianqihoubao.com/lishi/%s/month/%d%02d.html' % (city,year,month)

        headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.98 Safari/537.36 LBBROWSER'
        }
        r=''
        # 解决requests异常requests.exceptions.ConnectionError: HTTPSConnectionPool Max retries exceeded
        while r=='':
            try:
                r = requests.get(url,headers=headers)
            except Exception as e:
                time.sleep(5)
                print('is zzZZZZ...',url,e)
                continue
        # r.encoding = 'utf-8'  #
        text = r.content.decode("gbk", "ignore")
        html = etree.HTML(text)
        data_date  = html.xpath("//table[@class='b']//tr/td/a/text()")
        res = []
        for i in range(len(data_date)):
            date_ = html.xpath("//table[@class='b']//tr["+str(i+2)+"]/td/a/text()")
            weathe_ = html.xpath("//table[@class='b']//tr["+str(i+2)+"]/td/text()")
            #处理日期
            date_ = str(date_[0]).strip().replace('年','/').replace('月','/').replace('日','')
            #处理天气:消除字段中的空符号，并且把相同的天气类型合并
            weathe_w = [s.strip() for s in str(weathe_[2]).strip().split('/')]
            if weathe_w[0]==weathe_w[1]:
                weathe_w = weathe_w[0]
            else:
                weathe_w = '转'.join(weathe_w)
            #处理温度:把最小值放在前面，最大值放在后面
            weathe_t = [s.replace('℃','').strip() for s in str(weathe_[3]).strip().split('/')]
            temp_min, temp_max = weathe_t[1], weathe_t[0]
            res.append(','.join([city, date_, weathe_w, temp_min, temp_max]))
        return '\n'.join(res)

    def get_yubao(self,city_dir,timestr):
        citys = ['changsha', 'zhuzhou', 'xiangtan', 'shaoyang', 'hengyang', 'yueyang', 'changde', 'zhangjiajie',
                 'yiyang1',
                 'chenzhou', 'yongzhou', 'huaihua', 'loudi', 'jishou']
        year = int(datetime.datetime.now().strftime('%Y'))
        month = int(datetime.datetime.now().strftime('%m'))
        res = []
        for city in citys:
            res.append(self._get_lishi(city, year, month))
            res.append(self._get_yubao_2(city))
        open(city_dir+'/天气预报15天_各市州_%s.csv'%(timestr), 'w', encoding='utf-8').write('\n'.join(res))

    def get_lishi(self,train_data_dir):
        # 不同天气网站的城市编码不一样
        citys = ['changsha', 'zhuzhou', 'xiangtan', 'shaoyang', 'hengyang', 'yueyang', 'changde', 'zhangjiajie',
                 'yiyang',
                 'chenzhou', 'yongzhou', 'huaihua', 'loudi', 'jishou']

        for city in citys:
            result = []
            years = [2015,2016,2017,2018,2019,2020]
            for Y in years:
                if Y == 2020:
                    now_month = int(datetime.datetime.now().strftime('%m'))
                    for M in range(now_month):
                        result.append(self._get_lishi(city, Y, M + 1))
                        print(Y, '/', M)
                else:
                    for M in range(12):
                        result.append(self._get_lishi(city, Y, M + 1))
                        print(Y,'/',M)
                text = '\n'.join(result)
            open(train_data_dir+'/天气历史_%s.csv' % (city), 'w', encoding='utf-8').write(text)

    def get_lishi_addition(self, train_data_dir, years = [2015,2016,2017,2018,2019,2020], months = [1,2,3,4,5,6,7,8,9,10,11,12]):
        # 不同天气网站的城市编码不一样
        citys = ['changsha', 'zhuzhou', 'xiangtan', 'shaoyang', 'hengyang', 'yueyang', 'changde', 'zhangjiajie',
                 'yiyang',
                 'chenzhou', 'yongzhou', 'huaihua', 'loudi', 'jishou']

        for city in citys:
            result = []
            for Y in years:
                now_month = int(datetime.datetime.now().strftime('%m'))
                for M in months:
                    result.append(self._get_lishi(city, Y, M))
                    print(city, '/',Y, '/', M,'天气气温增量爬虫')
                text = '\n'.join(result)
            open(train_data_dir+'/天气历史_%s.csv' % (city), 'a+', encoding='utf-8').write('\n'+text)

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
            hunan_day = lines[idx::(len(lines)//14)]
            # header:[地区，日期，星期，天气，最低温，最高温]
            hunan_city = 'hunan'
            hunan_date = str(hunan_day.iat[0,1])
            hunan_weathe = str(hunan_day[2].value_counts().index[0])
            hunan_min = str(round(sum(hunan_day[3]._values * self.citys_w), 2))
            hunan_max = str(round(sum(hunan_day[4]._values * self.citys_w), 2))
            res.append(','.join([hunan_city,hunan_date,hunan_weathe,hunan_min,hunan_max]))
        open(predict_file, 'w', encoding='utf-8').write('\n'.join(res))
        return predict_file
    def get_feature(self,csv_file='./天气预报15天_湖南省.csv',train_data = './train_data/train_data.csv'):
        data_csv = pd.read_csv(csv_file, header=None,sep=',')
        train_data = pd.read_csv(train_data, sep=',')
        years = [2015, 2016, 2017, 2018, 2019, 2020]
        months = [1,2,3,4,5,6,7,8,9,10,11,12]
        weath_all = ['雷阵雨']
        for w in list(train_data[train_data.columns[2]]):
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

    def get_feature_lstm(self,csv_file='./天气预报15天_湖南省.csv',train_data = './train_data/train_data.csv'):
        data_csv = pd.read_csv(csv_file, header=None, sep=',')
        train_data = pd.read_csv(train_data, sep=',')

        years = [2015, 2016, 2017, 2018, 2019, 2020]
        months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        weath_all = []
        for w in list(train_data[train_data.columns[2]]):
            weath_all = weath_all + re.split('[转~]', w.strip())
        weath_all = list(set(list(weath_all)))
        weath_all = pinyin_sort(weath_all)
        print(weath_all)

        def cut_data(data_x, data_y):
            pass
            res_x = []
            res_y = []
            for i in range(len(data_x)):
                if any(data_x[i][0][6:11]):
                    res_x.append(data_x[i])
                    res_y.append(data_y[i])
            return np.array(res_x), np.array(res_y)

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
                if w == '雨': w = '小雨'
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
    def __init__(self, train_data_dir = './train_data'):
        self.pred_days = 15
        self.train_data_dir = train_data_dir
        self.citys_w = np.array([0.23,0.07,0.07,0.05,0.1,0.08,0.07,0.02,0.05,0.06,0.05,0.06,0.07,0.03])
    def get_hunan_info(self):
        citys = ['changsha', 'zhuzhou', 'xiangtan', 'shaoyang', 'hengyang', 'yueyang', 'changde', 'zhangjiajie',
                      'yiyang',
                      'chenzhou', 'yongzhou', 'huaihua', 'loudi', 'jishou']
        # if not os.path.exists(self.train_data_dir+'/各市州气候.csv'):
        txt_file = self.train_data_dir + '/天气历史_%s.csv' % (citys[0])
        data = pd.read_csv(txt_file, header=None, sep=',')
        for i in range(1, len(citys)):
            txt_file = self.train_data_dir + '/天气历史_%s.csv' % (citys[i])
            temp = pd.read_csv(txt_file, header=None, sep=',')
            data = pd.merge(data, temp, how='left', on=[data.columns[1], temp.columns[1]])
        # labels = pd.read_csv('label_2019.csv',header=None)
        # data=pd.merge(data, labels, how='inner', on=[data.columns[0], labels.columns[0]])
        data.to_csv(self.train_data_dir +'/各市州气候.csv', index=False, header=False)
        data = pd.read_csv(self.train_data_dir +'/各市州气候.csv', header=None, sep=',')
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
        open(self.train_data_dir +'/天气历史_湖南省.csv', 'w', encoding='utf-8').write('\n'.join(res))
        print('---')
    def get_feature(self,csv_file='./天气历史_湖南省.csv',label_file='./湖南省统调最大负荷.csv', train_data = './train_data.csv'):
        data_csv = pd.read_csv(csv_file, sep=',', names=['hunan', 'DATA_DATETIME', 'weather', 'tp_min', 'tp_max'])
        # hunan, DATA_DATETIME, weather, tp_min, tp_max
        data_label = pd.read_csv(label_file,sep=',')
        # 删除所有存在nan值的行再合并
        data_csv.drop(data_csv[np.isnan(data_csv["tp_max"])].index, inplace=True)
        data_ = pd.merge(data_csv, data_label, how='inner', on='DATA_DATETIME')
        data_['PARAM_VALUE'] = data_['PARAM_VALUE'] / 10000
        # 训练数据保存到文件
        data_.to_csv(train_data, index=False)
        # assert len(data_csv)==len(data_label)

        years = [2015, 2016, 2017, 2018, 2019, 2020]
        months = [1,2,3,4,5,6,7,8,9,10,11,12]
        weath_all = ['雷阵雨']
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
    pass
    replite = Replite()
    replite.get_lishi_addition(years=[2020], months=[8, 9])
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