#!/usr/bin/python3 -tt
'''------------------------------------- >Last Modified on Sun, 29 Jan 2017< '''
'''

'''

from collections import defaultdict
import numpy as np
import scipy.stats as stats
import scipy.sparse as sp
import itertools
import datetime
import pandas as pd
import binarycf as bcf

from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR 
#from xgboost import XGBRegressor

##################################################################

def rmse(u,v):
    return(np.sqrt(np.sum((u-v)**2)))

################################################################## decomp
df = pd.read_csv('df_long.csv')

cats = list(df.category.unique())
ks = [3,10] ## number of spectra
cv = 5

model_list = ['enet','rf','svm']
#model_list = ['enet','rf','svm','xgb']

param_enet = {'alpha': [0.01,0.1,1], 'l1_ratio': [0.1,0.3,1]}
param_rf = {'n_estimators': [10,30]}
param_svm = {'C': [0.01,0.1,1,10]}
param_xgb = {'n_estimators': [5,10], 'max_depth': [5,10],
             'reg_alpha': [0.01,0.1], 'reg_lambda': [0.01,0.1]}

df_result = pd.DataFrame(columns=['cat1','cat2','k','method','rmse_train',
                                  'score_train','rmse_cv','score_cv'])

np.random.seed(1)
for cat1, cat2, k in itertools.product(cats,cats,ks):
    if cat1 == cat2:
        continue

    print(cat1,cat2,k,datetime.datetime.now())

    df_common = pd.merge(bcf.customer_in_cat(df,cat1),
                         bcf.customer_in_cat(df,cat2),
                         on='customer')

    fold = np.random.permutation(np.mod(np.arange(df_common.shape[0]),cv))

    score_train = defaultdict(list) ## model -> list of training-score
    score_cv = defaultdict(list) ## model -> list of cv-score
    rmse_train = defaultdict(list) ## model -> list of rmse (on train)
    rmse_cv = defaultdict(list) ## model -> list of rmse (on train)

    for i in range(cv):

        X1, item_list1 = bcf.sets_to_matrix(df_common[cat1])
        X1_train = X1[fold!=i,:] ## sparse matrix
        y1_train = df_common[cat1][fold!=i] ## series of sets
        X1_test = X1[fold==i,:]
        y1_test = df_common[cat1][fold==i]
        A1_train, B1t, B1rinv = bcf.matrix_decomposition(X1_train,k=k)

        X2, item_list2 = bcf.sets_to_matrix(df_common[cat2])
        y2_train = df_common[cat2][fold!=i]
        X2_train = X2[fold!=i,:]
        X2_test = X2[fold==i,:]
        y2_test = df_common[cat2][fold==i]
        A2_train, B2t, B2rinv = bcf.matrix_decomposition(X2_train,k=k)

        k = A2_train.shape[1] ## 

        ########################################################## training

        models = defaultdict(list) ## model name -> list of models

        for j in range(k):
            grid_enet = GridSearchCV(ElasticNet(),param_enet,cv=3,
                                     scoring='mean_squared_error')
            grid_enet.fit(A1_train,A2_train[:,j])
            models['enet'].append(grid_enet)

            grid_rf = GridSearchCV(RandomForestRegressor(),param_rf,cv=3,
                                   scoring='mean_squared_error')
            grid_rf.fit(A1_train,A2_train[:,j])
            models['rf'].append(grid_rf)

            grid_svm = GridSearchCV(SVR(),param_svm,cv=3,
                                    scoring='mean_squared_error')
            grid_svm.fit(A1_train,A2_train[:,j])
            models['svm'].append(grid_svm)

            #grid_xgb = GridSearchCV(XGBRegressor(),param_xgb,cv=3,
            #                        scoring='mean_squared_error')
            #grid_xgb.fit(A1_train,A2_train[:,j])
            #models['xgb'].append(grid_xgb)

        ################################################## make a prediction

        A1_test = np.dot(X1_test.toarray(),B1rinv)
        A2_test = np.dot(X2_test.toarray(),B2rinv)

        for model in model_list:

            ## on train

            A2_train_hat = np.zeros(A2_train.shape)
            for j in range(k):
                A2_train_hat[:,j] = models[model][j].predict(A1_train)
            rmse_train[model].append(rmse(A2_train_hat,A2_train))

            X2_train_hat = np.dot(A2_train_hat,B2t)
            recom_ix = X2_train_hat.argsort(axis=1)[:,-5:]

            yhat = []
            for i in range(X2_train_hat.shape[0]):
                recom_items = set([item_list2[ix] for ix in recom_ix[i,:]])
                yhat.append(recom_items)
            yhat = pd.Series(yhat,index=y2_train.index)
            score_train[model].append(bcf.success_rate(yhat,y2_train))

            ## on test

            A2_test_hat = np.zeros(A2_test.shape)
            for j in range(k):
                A2_test_hat[:,j] = models[model][j].predict(A1_test)
            rmse_cv[model].append(rmse(A2_test_hat,A2_test))

            X2_test_hat = np.dot(A2_test_hat,B2t)
            recom_ix = X2_test_hat.argsort(axis=1)[:,-5:]

            yhat = []
            for i in range(X2_test_hat.shape[0]):
                recom_items = set([item_list2[ix] for ix in recom_ix[i,:]])
                yhat.append(recom_items)
            yhat = pd.Series(yhat,index=y2_test.index)
            score_cv[model].append(bcf.success_rate(yhat,y2_test))

    n = len(model_list)
    result = pd.DataFrame({
        'cat1'       : [cat1]*n,
        'cat2'       : [cat2]*n,
        'k'          : [k]*n,
        'method'     : model_list,
        'rmse_train' : [np.mean(rmse_train[model]) for model in model_list],
        'score_train': [np.mean(score_train[model]) for model in model_list],
        'rmse_cv'    : [np.mean(rmse_cv[model]) for model in model_list],
        'score_cv'   : [np.mean(score_cv[model]) for model in model_list]
    })
    print(result)
    df_result = pd.concat([df_result,result],axis=0)
    df_result.to_csv('df_results_svd.csv',index=False)


