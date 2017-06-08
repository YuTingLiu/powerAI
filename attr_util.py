# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 12:55:08 2017

@author: L
@attr filter
"""
from sklearn.datasets import load_boston
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
import json
import sys
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.svm import SVR
from sklearn import linear_model as lm 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from gbdt import *
class model_fac:
    def __init__(self):
        self.name = "create models for train"
    
    def model_set(self):
        gbdt = GradientBoostingRegressor()
        rf = RandomForestRegressor()
        ab = AdaBoostRegressor()
        svr = SVR()
        lm1 = lm.BayesianRidge()
        modelset = {}
        modelset['gbdt'] = gbdt
        modelset['rf'] = rf
        modelset['ab'] = ab
        modelset['svr'] = svr
        modelset['lm1'] = lm1
        return modelset
def data_fold(x_data,y_data,n_splits=3):
    '''input : N*D N*1 array
       output : TRAIN TEST LIST'''
    folded = []
    kf = KFold(n_splits=n_splits)
    for train_index, test_index in kf.split(x_data):
        print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = x_data[train_index], x_data[test_index]
        y_train, y_test = y_data[train_index], y_data[test_index]
        temp = {}
        y_train.shape = (len(y_train),1)
        y_test.shape = (len(y_test),1)
        temp['train'] = np.hstack((X_train,y_train))
        temp['test'] = np.hstack((X_test,y_test))
        folded.append(temp)
    return folded

def step1(folded,**model):
    if len(folded) != len(model):
        print('model number is ',len(model),'split dataset for ',len(folded))
        return 0
    keys = model.keys()
    keys = [x for x in keys]
    step1_result = [] 
    for i in range(len(folded)):
        train = folded[i]['train']
        x_train = train[:,:-1]
        y_train = train[:,-1]#lie vec  [:,-1] hang vec
        #print('x',x_train)
        #print('y',y_train)
        #train
        model[keys[i]].fit(x_train,y_train)
        #test
        test = folded[i]['test']
        x_test = test[:,:-1]
        y_test = test[:,-1]#lie vec  [:,-1] hang vec
        pred = model[keys[i]].predict(x_test)
        #print(y_test)
        #print(pred)
        step1_result.append(np.stack((y_test,pred),axis=0))
        #break
    return step1_result
#from minepy import MINE
#np.random.seed(0)
#size = 750
#X = np.random.uniform(0, 1, (size, 14))
##"Friedamn #1” regression problem
#Y = (10 * np.sin(np.pi*X[:,0]*X[:,1]) + 20*(X[:,2] - .5)**2 +
#    10*X[:,3] + 5*X[:,4] + np.random.normal(0,1))
##Add 3 additional correlated variables (correlated with X1-X3)
#X[:,10:] = X[:,:4] + np.random.normal(0, .025, (size,4))
#names = ["x%s" % i for i in range(1,15)]
class attr_filter:
    def __init__(self):
        self.name = 'fit data and evalute data attr '
        self.modelset = {}
        
    def models(self):
        gbdt = kdd_gbdt()
        rf = RandomForestRegressor()
        ab = AdaBoostRegressor()
#        svr = SVR()
        lm1 = lm.BayesianRidge()
        lr = LinearRegression(normalize=True)
        ridge = Ridge(alpha=7)
        lasso = Lasso(alpha=.05)
        rlasso = RandomizedLasso(alpha=0.04)
        modelset = {}
        modelset['gbdt'] = gbdt
        modelset['rf'] = rf
        modelset['ab'] = ab
#        modelset['svr'] = svr
        modelset['lm1'] = lm1
        modelset['lr'] = lr
        modelset['ridge'] = ridge
        modelset['lasso'] = lasso
        modelset['rlasso'] = rlasso
        self.modelset = modelset
        
    def rank_attr(self,names,X,Y):
        '''
        不同分类器对特征进行排序
        '''
        self.models()
        
        keys = self.modelset.keys()
        keys = [x for x in keys]
        
        if len(names) != len(X[0]):
            print('columns error',names)
            return 0
        
        ranks = {}
        def rank_to_dict(ranks, names, order=1):
            minmax = MinMaxScaler()
            ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
            ranks = map(lambda x: round(x, 2), ranks)
            return dict(zip(names, ranks ))
        for i in range(len(keys)):
            model = self.modelset[keys[i]]
            model.fit(X, Y)
            try:
                ranks[keys[i]] = rank_to_dict(np.abs(model.coef_), names)
            except:
                try:
                    ranks[keys[i]] = rank_to_dict(np.abs(model.scores_), names)
                except:
                    try:
                        ranks[keys[i]] = rank_to_dict(np.abs(model.feature_importances_), names)
                    except:
                        try:
                            ranks[keys[i]] = rank_to_dict(np.abs(model.model.feature_importances_), names)
                        except:
                            pass
        #mine = MINE()
        #mic_scores = []
        #for i in range(X.shape[1]):
        #    mine.compute_score(X[:,i], Y)
        #    m = mine.mic()
        #    mic_scores.append(m)
        #ranks["MIC"] = rank_to_dict(mic_scores, names)
        r = {}
        for name in names:
            r[name] = round(np.mean([ranks[method][name] for method in ranks.keys()]), 2)
        methods = sorted(ranks.keys())
        ranks["Mean"] = r
        methods.append("Mean")
        print ("   \t%s" % "\t".join(methods))
        for name in names:
            print("%s  \t %s " % (name, "\t".join(map(str, 
                                [ranks[method][name] for method in methods]))))
        rankdf = pd.DataFrame.from_dict(ranks)
        print(rankdf)
        rankdf.to_csv('attr_rank.csv')

def main():
    f = open('columns.txt','r')
    attr = json.load(f)
    f.close()
    print(attr)
    f = open('one_hot_code.txt','r')
    hot_code = json.load(f).split(',')
    f.close()
    print(hot_code)
    for i in range(1,1455):
        df = pd.read_csv(''.join([r'../train/user/',str(i),'.csv']))
        df.loc[:,'record_date']=pd.to_datetime(df['record_date'],format=r'%Y/%m/%d %H:%M:%S')
        df = df.dropna()
#        x1 = df[attr]
        x = df[hot_code]
        if np.any(pd.isnull(x)):
            print('nan found ')
            print(x[np.sum(x.isnull(),axis=1)>0])
            sys.exit()
        y = df['power_consumption']
        # 随机抽取20%的测试集
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        print(x_train.shape,y_train.shape)
        print(x_test.shape,y_test.shape)
        print(list(x_train.columns))
        model = attr_filter()
        model.rank_attr(list(x_train.columns),np.array(x_test),np.array(y_test))
        sys.exit()
        
main()