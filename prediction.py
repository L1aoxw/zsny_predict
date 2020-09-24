# -*- conding:utf-8 -*-
from replite import Merge_train,Merge_pred
import os
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_absolute_error
from xgboost import XGBClassifier,XGBRegressor
from xgboost import plot_importance
from sklearn.model_selection import GridSearchCV   #Perforing grid search
import xgboost as xgb
import pandas as pd
import datetime
import numpy as np

class Prediction_xgb:
    def __init__(self, model_file):
        self.xgb_model_path = model_file
        self.param = {
            'learning_rate':0.1,
            'max_depth': 10,
            'gamma': 0,
            'subsample': 1,
            'colsample': 0.75,
            'colsample_bytree': 0.75,
            'scale_pos_weight': 1,
            'verbosity': 3,
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
        }
        self.model = XGBRegressor(
            slice=1,
            learning_rate=0.1,
            n_estimators=96,  # 树的个数--1000棵树建立xgboost
            max_depth=5,  # 树的深度
            min_child_weight=2,  # 叶子节点最小权重
            gamma=0.,  # 惩罚项中叶子结点个数前的参数
            subsample=0.8,  # 随机选择80%样本建立决策树
            # colsample = 0.75,  # 随机选择80%特征建立决策树
            colsample_bytree=0.6,
            verbosity=3,
            objective='reg:squarederror',  # 指定损失函数
            eval_metric='mae',
            scale_pos_weight=1,  # 解决样本个数不平衡的问题
            random_state = 27,  # 随机数
            )

    def cut_data(self,data_x,data_y):
        pass
        res_x = []
        res_y = []
        for i in range(data_x.shape[0]):
            if any(data_x[i][4:11]):
                res_x.append(data_x[i])
                res_y.append(data_y[i])
        return np.array(res_x),np.array(res_y)

    def train_XGBClassifier(self):
        print('---xgb start---')
        self.model.fit(self.train_data,self.train_y,eval_set=[(self.train_data,self.train_y)])

    def train_xgboost(self,trian_data, train_y):
        print('---xgb start---')
        train_data, train_y = self.cut_data(trian_data, train_y)
        train_xdf = pd.DataFrame(train_data[:-50])
        train_ydf = pd.DataFrame(train_y[:-50])
        test_xdf = pd.DataFrame(train_data[-50:])
        test_ydf= pd.DataFrame(train_y[-50:])

        dtrain = xgb.DMatrix(train_xdf, label=train_ydf)
        dtest = xgb.DMatrix(test_xdf, label=test_ydf)

        def modelfit(alg, train_xdf,train_ydf, useTrainCV=True, cv_folds=5, early_stopping_rounds=20):
            if useTrainCV:
                xgb_param = alg.get_xgb_params()
                xgtrain = xgb.DMatrix(train_xdf, label=train_ydf)
                cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                                  metrics='mae', early_stopping_rounds=early_stopping_rounds)

                alg.set_params(n_estimators=cvresult.shape[0])
                print(cvresult.shape[0])

            # Fit the algorithm on the data
            alg.fit(train_xdf, train_ydf, eval_metric='mae')
            # bst = xgb.train(alg.get_xgb_params(), xgtrain, num_boost_round=cvresult.shape[0])

            # Predict training set:
            dtrain_predictions = alg.predict(test_xdf)
            # dtrain_predictions = bst.predict(dtest)

            # Print model report:
            print("\nModel Report")
            test_y = dtest.get_label()
            print('error---', np.mean(abs(dtrain_predictions - test_y) / test_y))
            print("Accuracy : %.4g" % mean_absolute_error(test_y, dtrain_predictions))
            # print("Accuracy : %.4g" % np.mean(abs(dtrain_predictions - np.array(train_ydf[0])) / np.array(train_ydf[0])))

            feat_imp = pd.Series(alg.get_booster().get_fscore()).sort_values(ascending=False)
            feat_imp.plot(kind='bar', title='Feature Importances')
            plt.ylabel('Feature Importance Score')
            plt.show()

        # modelfit(self.model, train_xdf, train_ydf)

        param_test1 = {
            # 'max_depth': range(4, 7, 1),
            # 'min_child_weight': range(2, 4, 1)
            # 'gamma': [i / 10.0 for i in range(0, 5)]
            'subsample': [i / 10.0 for i in range(6, 10)],
            'colsample_bytree': [i / 10.0 for i in range(6, 10)]
        }
        gsearch1 = GridSearchCV(
            estimator=XGBRegressor(
                                slice=1,
                                learning_rate= 0.01,
                                n_estimators = 96,  # 树的个数--1000棵树建立xgboost
                                max_depth = 5,  # 树的深度
                                min_child_weight = 2,  # 叶子节点最小权重
                                gamma = 0.,  # 惩罚项中叶子结点个数前的参数
                                subsample = 0.8,  # 随机选择80%样本建立决策树
                                # colsample = 0.75,  # 随机选择80%特征建立决策树
                                colsample_bytree = 0.6,
                                verbosity = 3,
                                objective = 'reg:squarederror',  # 指定损失函数
                                eval_metric='mae',
                                scale_pos_weight = 1,  # 解决样本个数不平衡的问题
                                # random_state = 27,  # 随机数
                                ),
            param_grid=param_test1, scoring='neg_mean_absolute_error', n_jobs=4, iid=False, cv=5)
        # gsearch1.fit(test_xdf, test_ydf)
        # print(gsearch1.cv_results_, gsearch1.best_params_, gsearch1.best_score_)

        bst = xgb.train(self.param,dtrain,num_boost_round=200)
        bst.save_model(self.xgb_model_path)
        test_preds = bst.predict(dtest)  #
        test_y = dtest.get_label()
        print('error---',np.mean(abs(test_preds - test_y) / test_y))
        print('---')
        self.model.fit(train_xdf, train_ydf, eval_metric='mae')
        # bst = xgb.train(alg.get_xgb_params(), xgtrain, num_boost_round=cvresult.shape[0])

        # Predict training set:
        dtrain_predictions = self.model.predict(test_xdf)
        # dtrain_predictions = bst.predict(dtest)

        # Print model report:
        print("\nModel Report")
        test_y = dtest.get_label()
        print('error---', np.mean(abs(dtrain_predictions - test_y) / test_y))
        print("Accuracy : %.4g" % mean_absolute_error(test_y, dtrain_predictions))
        feat_imp = pd.Series(self.model.get_booster().get_fscore()).sort_values(ascending=False)
        feat_imp.plot(kind='bar', title='Feature Importances')
        plt.ylabel('Feature Importance Score')
        plt.show()
        pass

    def pred_xgboost(self,predict_data):
        pass
        predict_df = pd.DataFrame(predict_data)
        dpred = xgb.DMatrix(predict_df)
        bst = xgb.Booster(model_file=self.xgb_model_path)
        preds = bst.predict(dpred)*1.05
        print('preds---', preds)
        print('---')
        return preds
        pass

if __name__ == '__main__':
    merge_train = Merge_train()
    merge_pred = Merge_pred()

    timestr = datetime.datetime.now().strftime('%Y-%m-%d-%H')
    predict_dir = './data'
    predict_file = predict_dir + '/天气预报15天_湖南省_%s.csv' % (timestr)
    # replite.get_lishi()
    if not os.path.exists('./天气历史_湖南省.csv'):
        merge_train.get_hunan_info()
    if not os.path.exists('./天气预报15天_湖南省_%s.csv'%(timestr)):
        merge_pred.get_hunan_info()
    trian_data, train_y = merge_train.get_feature()
    # merge_pred.get_hunan_info()
    predict_data = merge_pred.get_feature()

    prediction = Prediction_xgb()
    prediction.train_xgboost(trian_data, train_y)
    preds = prediction.pred_xgboost(predict_data)

    print('---')