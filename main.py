# -*- conding:utf-8 -*-
from orcaleDB import oracleOperation
from prediction import Prediction_xgb
from ai_model import tf_predict
from replite import Replite,Merge_train,Merge_pred
from Logger import Logger
from sklearn import metrics   #Additional     scklearn functions
from sklearn.model_selection import GridSearchCV   #Perforing grid search
import tensorflow as tf
import numpy as np

import os
import datetime
import uuid
import pandas as pd
import time
import ai_model
import importlib,sys
importlib.reload(sys)
RD_ID = {
    'hunan':'430000',
    'changsha':'430100',
    'zhuzhou':'430200',
    'xiangtan':'430300',
    'hengyang':'430400',
    'shaoyang':'430500',
    'yueyang':'430600',
    'changde':'430700',
    'zhangjiajie':'430800',
    'yiyang1':'430900',
    'chenzhou':'431000',
    'yongzhou':'431100',
    'huaihua':'431200',
    'loudi':'431300',
    'jishou':'433100',
    }

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
pass
def insert_weather(db, city_temp, hunan_temp, value_idx='max'):
    connection = db.openOracleConn()
    date_now = datetime.datetime.now()
    insertParams = []
    updateParams = []

    assert value_idx in ['max','min']
    if value_idx == 'max': param_paramid = 'PA43000644*'
    elif value_idx == 'min': param_paramid = 'PA43000643*'

    lines = open(hunan_temp, 'r', encoding='utf-8').readlines()+open(city_temp, 'r', encoding='utf-8').readlines()
    for line in lines:
        words = line.strip().split(',')
        detester = words[1]
        data_datetime = datetime.datetime.strptime(detester,'%Y/%m/%d')
        # 只入库当前日期之后的十五天预测数据
        if data_datetime < datetime.datetime.now(): continue

        param_RD_ID2 = RD_ID[words[0]]
        if value_idx == 'max': param_value = words[-1]
        elif value_idx == 'min': param_value = words[-2]
        param_select = {
            # 'event_id':uuid.uuid4().hex,
            'table_id': 'TA43070001',
            'edition_id': 'DEFAULT',
            'source_id': 'S430005',
            'param_id': param_paramid,
            'timeflag_id': 'D',
            'data_datetime': data_datetime,
            'rd_id1': 'R43002001',
            'rd_id2': param_RD_ID2,
            # 'param_value':param_value,
            'param_unit': 'U430134',
            # 'input_datetime':date_now,
            # 'last_modified':date_now,
            'data_type': '1'
        }
        if db.select_C_TR2P_VALUES(connection, param_select):
            param_update = param_select
            param_update['param_value'] = param_value
            param_update['last_modified'] = date_now
            param_update['TRANS_HNY_STATUS'] = ''   #为空未迁移，1已迁移
            param_update['TRANS_STATUS_NYJ'] = ''   #为空未迁移，1已迁移
            param_update['TRANS_STATUS_SELF'] = ''  #为空未迁移，1已迁移
            updateParams.append(param_update)
        else:
            param_insert = param_select
            param_insert['event_id'] = uuid.uuid4().hex
            param_insert['param_value'] = param_value
            param_insert['input_datetime'] = date_now
            param_insert['last_modified'] = date_now
            insertParams.append(param_insert)

    db.update_C_TR2P_VALUES(connection, updateParams)
    db.insert_C_TR2P_VALUES(connection, insertParams)
    connection.close()
    pass
def insert_preds(db,save_file:str,cate='最大负荷'):
    # cate:最大负荷、日用电量、日用气量
    connection = db.openOracleConn()
    date_now = datetime.datetime.now()
    insertParams = []
    updateParams = []
    for line in open(save_file,'r',encoding='utf-8').readlines():
        words = line.strip().split(',')

        detester = words[1]
        data_datetime = datetime.datetime.strptime(detester,'%Y/%m/%d')
        # 只入库当前日期之后的十五天预测数据
        if data_datetime < datetime.datetime.now() : continue
        param_value = words[-1]
        if cate == '最大负荷':
            param_select = {
                # 'event_id':uuid.uuid4().hex,
                'table_id': 'TA43030001',
                'edition_id': 'DEFAULT',
                'source_id': 'S430018',
                'param_id': 'PA43000230*',
                'timeflag_id': 'D',
                'data_datetime': data_datetime,
                'rd_id1': 'R43014022',
                'rd_id2': '430000',
                # 'param_value':param_value,
                'param_unit': 'U430019',
                # 'input_datetime':date_now,
                # 'last_modified':date_now,
                'data_type': '1'
            }
        elif cate == '日用电量':
            param_select = {
                # 'event_id':uuid.uuid4().hex,
                'table_id': 'TA43030001',
                'edition_id': 'DEFAULT',
                'source_id': 'S430018',
                'param_id': 'PA43000362*',
                'timeflag_id': 'D',
                'data_datetime': data_datetime,
                'rd_id1': 'R43014022',
                'rd_id2': '430000',
                # 'param_value':param_value,
                'param_unit': 'U430136',
                # 'input_datetime':date_now,
                # 'last_modified':date_now,
                'data_type': '1'
            }
        if db.select_C_TR2P_VALUES(connection,param_select):
            param_update = param_select
            param_update['param_value'] = param_value
            param_update['last_modified'] = date_now
            param_update['TRANS_HNY_STATUS'] = ''   #为空未迁移，1已迁移
            param_update['TRANS_STATUS_NYJ'] = ''   #为空未迁移，1已迁移
            param_update['TRANS_STATUS_SELF'] = ''  #为空未迁移，1已迁移
            updateParams.append(param_update)
        else:
            param_insert = param_select
            param_insert['event_id'] = uuid.uuid4().hex
            param_insert['param_value'] = param_value
            param_insert['input_datetime'] = date_now
            param_insert['last_modified'] = date_now
            insertParams.append(param_insert)

    db.update_C_TR2P_VALUES(connection, updateParams)
    db.insert_C_TR2P_VALUES(connection, insertParams)
    connection.close()
    pass

def train_model(train_data_dir,model_file):
    # replite = Replite()
    # replite.get_lishi_addition(train_data_dir,years=[2020], months=[8])

    merge_train = Merge_train(train_data_dir)
    merge_pred = Merge_pred()
    print('------模型重新训练------')
    if not os.path.exists('./train_data/天气历史_湖南省.csv'):
        merge_train.get_hunan_info()
    predict_dir = './data'
    # predict_file = predict_dir + '/天气预报15天_湖南省_%s.csv' % (timestr)
    csv_flie = train_data_dir + '/天气历史_湖南省.csv'
    label_file = train_data_dir +'/湖南省实际用电量.csv'
    train_file = train_data_dir +'/train_data.csv'
    trian_data, train_y = merge_train.get_feature(csv_flie,label_file,train_file)
    # merge_pred.get_hunan_info()
    prediction = Prediction_xgb(model_file)
    prediction.train_xgboost(trian_data, train_y)
    # predict_data = merge_pred.get_feature(predict_file,train_file)
    # preds = prediction.pred_xgboost(predict_data)
    # print(preds)
    print('------训练完成------')

def predict_weather():
    db = oracleOperation()
    merge_pred = Merge_pred()

    timestr = datetime.datetime.now().strftime('%Y-%m-%d-%H')
    data_dir = './data'
    city_dir = './data_city'
    city_data = city_dir + '/天气预报15天_各市州_%s.csv' % (timestr)
    hunan_data = data_dir + '/天气预报15天_湖南省_%s.csv' % (timestr)
    if not os.path.exists(hunan_data):
        print(hunan_data, 'not exist')
        predict_file = merge_pred.get_hunan_info(hunan_data, timestr)
    print('插入天气预报最高气温入库：')
    insert_weather(db, city_data, hunan_data, 'max')
    print('插入天气预报最低气温入库：')
    insert_weather(db, city_data, hunan_data, 'min')

def predict_lstm_ydl():
    result_dir = './result.ydl/lstm'
    data_dir = './data'
    city_dir = './data_city'

    train_data_dir = './train_data'
    # model_file = './xgb_model/xgb_model_file'
    model_file_ydl = './xgb_model/xgb_model_ydl'
    label_file = './湖南省全省用电量.csv'
    timestr = datetime.datetime.now().strftime('%Y-%m-%d-%H')
    save_file = result_dir + '/result15天_湖南省_%s.csv' % (timestr)
    city_data = city_dir + '/天气预报15天_各市州_%s.csv' % (timestr)
    hunan_data = data_dir + '/天气预报15天_湖南省_%s.csv' % (timestr)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.reset_default_graph()
    sess = tf.Session(config=config)
    sess_file =  model_path = os.path.join(PROJECT_DIR, "data.out.train/lstm_fdl/model.weights")

    print('---------------', time.strftime('%Y-%m-%d-%H:%M'), '---------------')
    db = oracleOperation()

    prediction = tf_predict.Predictor_fdl(sess, 'E:\git\zsny_predict\data.in.train\lstm_fdl\epoch=8000\model.ckpt')
    if not os.path.exists(save_file):

        merge_pred = Merge_pred()
        # replite.get_lishi()
        if not os.path.exists(hunan_data):
            print(hunan_data, 'not exist')
            predict_file = merge_pred.get_hunan_info(hunan_data, timestr)
        # trian_data, train_y = merge_train.get_feature()
        # prediction.train_xgboost(trian_data, train_y)
        # predict_file = merge_pred.get_hunan_info(timestr)
        predict_data = merge_pred.get_feature_lstm(hunan_data)

        print('---设置样本的时间序列步长---')
        predict_X = []
        for i in range(7, len(predict_data)):
            predict_X.append(predict_data[i - 7:i])
        preds = prediction.predict(np.array(predict_X), keep_prob = 1 )
        result_file = pd.read_csv(hunan_data, header=None, sep=',', skiprows=7)
        result_file.insert(len(result_file.columns), '', preds)
        result_file.to_csv(save_file, index=False, header=False)
        # return save_file

        # print('插入天气预报最高气温入库：')
        # insert_weather(db, city_data, hunan_data, 'max')
        # print('插入天气预报最低气温入库：')
        # insert_weather(db, city_data, hunan_data, 'min')
    print('插入用电量预测值入库：')
    insert_preds(db, save_file, cate='日用电量')
    print('----------------------------------------------------------------')

def predict_xgb_fh(is_train = False):
    result_dir = './result.fh/xgb'
    data_dir = './data'
    city_dir = './data_city'
    train_data_dir = './train_data'
    # model_file = './xgb_model/xgb_model_file'
    model_file_fh = './xgb_model/xgb_model_file'
    timestr = datetime.datetime.now().strftime('%Y-%m-%d-%H')
    save_file = result_dir + '/result15天_湖南省_%s.csv' % (timestr)
    city_data = city_dir + '/天气预报15天_各市州_%s.csv' % (timestr)
    hunan_data = data_dir + '/天气预报15天_湖南省_%s.csv' % (timestr)

    if is_train: train_model(train_data_dir, model_file_fh)

    print('---------------', time.strftime('%Y-%m-%d-%H:%M'), '---------------')
    db = oracleOperation()

    prediction = Prediction_xgb(model_file_fh)
    if not os.path.exists(save_file):
        print(save_file, ' not exist')
        # save_file = './result15天_湖南省_%s.csv' % (timestr)
        merge_pred = Merge_pred()
        # replite.get_lishi()
        if not os.path.exists(hunan_data):
            print(hunan_data, 'not exist')
            predict_file = merge_pred.get_hunan_info(hunan_data, timestr)
        # trian_data, train_y = merge_train.get_feature()
        # prediction.train_xgboost(trian_data, train_y)
        # predict_file = merge_pred.get_hunan_info(timestr)
        predict_data = merge_pred.get_feature(hunan_data)

        preds = prediction.pred_xgboost(predict_data)
        result_file = pd.read_csv(hunan_data, header=None, sep=',')
        result_file.insert(len(result_file.columns), '', preds)
        result_file.to_csv(save_file, index=False, header=False)

    # print('插入天气预报最高气温入库：')
    # insert_weather(db, city_data, hunan_data, 'max')
    # print('插入天气预报最低气温入库：')
    # insert_weather(db, city_data, hunan_data, 'min')
    print('插入负荷预测值入库：')
    insert_preds(db, save_file, cate='最大负荷')
    print('----------------------------------------------------------------')

def predict_xgb_ydl(is_train = False):
    result_dir = './result.ydl/xgb'
    data_dir = './data'
    city_dir = './data_city'
    train_data_dir = './train_data'
    # model_file = './xgb_model/xgb_model_file'
    model_file_ydl = './xgb_model/xgb_model_ydl'
    timestr = datetime.datetime.now().strftime('%Y-%m-%d-%H')
    save_file = result_dir + '/result15天_湖南省_%s.csv' % (timestr)
    city_data = city_dir + '/天气预报15天_各市州_%s.csv' % (timestr)
    hunan_data = data_dir + '/天气预报15天_湖南省_%s.csv' % (timestr)

    if is_train: train_model(train_data_dir , model_file_ydl)

    print('---------------', time.strftime('%Y-%m-%d-%H:%M'), '---------------')
    db = oracleOperation()
    merge_train = Merge_train(train_data_dir)
    prediction = Prediction_xgb(model_file_ydl)
    if not os.path.exists(save_file):
        print(save_file, ' not exist')
        # save_file = './result15天_湖南省_%s.csv' % (timestr)
        merge_pred = Merge_pred()
        # replite.get_lishi()
        if not os.path.exists(hunan_data):
            print(hunan_data, 'not exist')
            predict_file = merge_pred.get_hunan_info(hunan_data, timestr)
        # trian_data, train_y = merge_train.get_feature()
        # prediction.train_xgboost(trian_data, train_y)
        predict_data = merge_pred.get_feature(hunan_data)

        preds = prediction.pred_xgboost(predict_data)*1.05*10000
        result_file = pd.read_csv(hunan_data, header=None, sep=',')
        result_file.insert(len(result_file.columns), '', preds)
        result_file.to_csv(save_file, index=False, header=False)

    # print('插入天气预报最高气温入库：')
    # insert_weather(db, city_data, hunan_data, 'max')
    # print('插入天气预报最低气温入库：')
    # insert_weather(db, city_data, hunan_data, 'min')
    print('插入日用电量预测值入库：')
    insert_preds(db, save_file, cate='日用电量')
    print('----------------------------------------------------------------')


if __name__ == '__main__':

    sys.stdout = Logger(sys.stdout)  # 将输出记录到log
    sys.stderr = Logger(sys.stderr)  # 将错误信息记录到log
    predict_weather()
    predict_xgb_fh()
    # predict_lstm_ydl()
    predict_xgb_ydl()