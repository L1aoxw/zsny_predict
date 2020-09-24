# -*- conding:utf-8 -*-
from orcaleDB import oracleOperation
from prediction import Prediction_xgb
from replite import Replite,Merge_train,Merge_pred

from sklearn import metrics   #Additional     scklearn functions
from sklearn.model_selection import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams

import os
import datetime
import uuid
import pandas as pd
import time
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

class Logger(object):

    def __init__(self, stream=sys.stdout):
        output_dir = "./log"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H'))
        filename = os.path.join(output_dir, log_name)

        self.terminal = stream
        self.log = open(filename, 'a+',encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
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
            param_update['TRANS_HNY_STATUS'] = '1'
            param_update['TRANS_STATUS_NYJ'] = '1'
            param_update['TRANS_STATUS_SELF'] = '1'
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

def insert_preds(db,save_file):
    connection = db.openOracleConn()
    date_now = datetime.datetime.now()
    insertParams = []
    updateParams = []
    for line in open(save_file,'r',encoding='utf-8').readlines():
        words = line.strip().split(',')

        detester = words[1]
        data_datetime = datetime.datetime.strptime(detester,'%Y/%m/%d')
        param_value = words[-1]
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
        if db.select_C_TR2P_VALUES(connection,param_select):
            param_update = param_select
            param_update['param_value'] = param_value
            param_update['last_modified'] = date_now
            param_update['TRANS_HNY_STATUS'] = '1'
            param_update['TRANS_STATUS_NYJ'] = '1'
            param_update['TRANS_STATUS_SELF'] = '1'
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

def predict_result(timestr,save_file):
    predict_dir = './data'
    predict_file = predict_dir+'/天气预报15天_湖南省_%s.csv' % (timestr)
    # save_file = './result15天_湖南省_%s.csv' % (timestr)
    merge_pred = Merge_pred()
    # replite.get_lishi()
    if not os.path.exists(predict_file):
        print(predict_file,'not exist')
        predict_file = merge_pred.get_hunan_info(predict_file,timestr)
    # trian_data, train_y = merge_train.get_feature()
    # prediction.train_xgboost(trian_data, train_y)
    # predict_file = merge_pred.get_hunan_info(timestr)
    predict_data = merge_pred.get_feature(predict_file)

    preds = prediction.pred_xgboost(predict_data)
    result_file = pd.read_csv(predict_file, header=None, sep=',')
    result_file.insert(len(result_file.columns), '', preds)
    result_file.to_csv(save_file, index=False, header=False)
    # return save_file

def train_model(train_data_dir,model_file):
    # replite = Replite()
    # replite.get_lishi_addition(train_data_dir,years=[2020], months=[8])

    def modelfit(alg, dtrain, predictors, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
        if useTrainCV:
            xgb_param = alg.get_xgb_params()
            xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
            cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                              metrics='auc', early_stopping_rounds=early_stopping_rounds, show_progress=False)
            alg.set_params(n_estimators=cvresult.shape[0])

        # Fit the algorithm on the data
        alg.fit(dtrain[predictors], dtrain['Disbursed'], eval_metric='auc')

        # Predict training set:
        dtrain_predictions = alg.predict(dtrain[predictors])
        dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

        # Print model report:
        print("Model Report")
        print
        "Accuracy : %.4g" % metrics.accuracy_score(dtrain['Disbursed'].values, dtrain_predictions)
        print
        "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain['Disbursed'], dtrain_predprob)

        feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')

    merge_train = Merge_train(train_data_dir)
    merge_pred = Merge_pred()
    print('------模型重新训练------')
    if not os.path.exists('./train_data/天气历史_湖南省.csv'):
        merge_train.get_hunan_info()
    predict_dir = './data'
    predict_file = predict_dir + '/天气预报15天_湖南省_%s.csv' % (timestr)
    csv_flie = train_data_dir + '/天气历史_湖南省.csv'
    label_file = train_data_dir +'/201611-202008湖南省实际用电量.csv'
    train_file = train_data_dir +'/train_data.csv'
    trian_data, train_y = merge_train.get_feature(csv_flie,label_file,train_file)
    # merge_pred.get_hunan_info()
    prediction = Prediction_xgb(model_file)
    prediction.train_xgboost(trian_data, train_y)
    # predict_data = merge_pred.get_feature(predict_file,train_file)
    # preds = prediction.pred_xgboost(predict_data)
    # print(preds)
    print('------训练完成------')

if __name__ == '__main__':

    result_dir = './result'
    data_dir = './data'
    city_dir = './data_city'
    train_data_dir = './train_data'
    # model_file = './xgb_model/xgb_model_file'
    model_file_ydl = './xgb_model/xgb_model_ydl'
    train_flag = True
    label_file = './201611-202008湖南省统调最大负荷.csv'
    timestr = datetime.datetime.now().strftime('%Y-%m-%d-%H')
    save_file = result_dir + '/result15天_湖南省_%s.csv' % (timestr)
    city_data = city_dir + '/天气预报15天_各市州_%s.csv' % (timestr)
    hunan_data = data_dir + '/天气预报15天_湖南省_%s.csv' % (timestr)

    sys.stdout = Logger(sys.stdout) #  将输出记录到log
    sys.stderr = Logger(sys.stderr)   # 将错误信息记录到log
    print('---------------', time.strftime('%Y-%m-%d-%H:%M'), '---------------')
    db = oracleOperation()

    if train_flag: train_model(train_data_dir,model_file_ydl)

    prediction = Prediction_xgb(model_file_ydl)
    if not os.path.exists(save_file):
        print(save_file,' not exist')
        predict_result(timestr,save_file)

    print('插入天气预报最高气温入库：')
    insert_weather(db, city_data, hunan_data, 'max')
    print('插入天气预报最低气温入库：')
    insert_weather(db, city_data, hunan_data, 'min')
    print('插入负荷预测值入库：')
    insert_preds(db,save_file)
    print('----------------------------------------------------------------')