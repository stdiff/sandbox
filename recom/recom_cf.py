#!/usr/bin/python3 -tt
'''------------------------------------- >Last Modified on Sun, 29 Jan 2017< '''
'''

real    1044m34.972s
user    1041m22.459s
sys     2m34.797s

'''

from collections import defaultdict 
import itertools
import datetime
import numpy as np
import pandas as pd
import binarycf as bcf

########################################################################

def insert_result(result,row_dict):
    result = result.copy()
    for key in result.keys():
        result[key].append(row_dict[key])
    return(result)

########################################################################

df_long = pd.read_csv('df_long.csv')

cats = list(df_long.category.unique())

cat_variety = df_long.groupby(['category','item']).size().reset_index()
cat_variety = cat_variety['category'].value_counts()

cats_df = dict() ## cat -> data frame of the cat 
## this takes a quite long time 
for cat in cats:
    cats_df[cat] = bcf.customer_in_cat(df_long,cat)

#######################################################################

cols = ['cat1','cat2','method','metric','k','train','cv']
df_result = dict()
for col in cols:
    df_result[col] = []

metrics = {'common': bcf.common_similarity,
           'cosine': bcf.cosine_similarity,
           #'correlation': bcf.correlation_similarity
       }

ks = [10,30,50]
cv = 5

for cat1, cat2 in itertools.product(cats,cats):
    if cat1 == cat2:
        continue
    print(cat1,cat2,str(datetime.datetime.now()))
    df_common = pd.merge(cats_df[cat1],cats_df[cat2],on='customer')

    np.random.seed(1)
    fold = np.random.permutation(np.mod(np.arange(df_common.shape[0]),cv))

    row_dict = {'cat1': cat1, 'cat2': cat2, 'metric': np.nan, 'k' : np.nan}

    ## popular 
    tr_popular = []
    cv_popular = []

    for i in range(cv):
        X_train = df_common[cat1][fold!=i]
        y_train = df_common[cat2][fold!=i]
        X_test = df_common[cat1][fold==i]
        y_test = df_common[cat2][fold==i]

        popular_items = bcf.get_popular_items(y_train)

        yhat = X_train.apply(lambda x: popular_items)
        tr_popular.append(bcf.success_rate(yhat,y_train))

        yhat = X_test.apply(lambda x: popular_items)
        cv_popular.append(bcf.success_rate(yhat,y_test))

    row_dict['method'] = 'popular'
    row_dict['train'] = np.mean(tr_popular)
    row_dict['cv'] = np.mean(cv_popular)
    df_result = insert_result(df_result,row_dict)


    ## collaborative filterings
    for metric, k in itertools.product(metrics.keys(),ks):
        print(metric,k,str(datetime.datetime.now()))
        row_dict['metric'] = metric
        row_dict['k'] = k
        met = metrics[metric] ## function

        tr_ibcf = []
        cv_ibcf = []
        tr_ubcf = []
        cv_ubcf = []

        for i in range(cv):
            df_common_train = df_common[fold!=i]
            X_train = df_common[cat1][fold!=i]
            y_train = df_common[cat2][fold!=i]

            X_test = df_common[cat1][fold==i]
            y_test = df_common[cat2][fold==i]

            ## IBCF
            if metric == 'correlation':
                met = bcf.correlation_similarity(X_train.shape[0])

            similar_ibcfs = bcf.compute_similar_items(df_common_train,met,k=k)

            yhat = X_train.apply(lambda x: bcf.ibcf(x,similar_ibcfs))
            tr_ibcf.append(bcf.success_rate(yhat,y_train))

            yhat = X_test.apply(lambda x: bcf.ibcf(x,similar_ibcfs))
            cv_ibcf.append(bcf.success_rate(yhat,y_test))

            ## UBCF
            if metric == 'correlation':
                met = bcf.correlation_similarity(cat_variety[cat1])

            yhat = X_train.apply(lambda x: bcf.ubcf(x,X_train,y_train,met,k=k))
            tr_ubcf.append(bcf.success_rate(yhat,y_train))

            yhat = X_test.apply(lambda x: bcf.ubcf(x,X_train,y_train,met,k=k))
            cv_ubcf.append(bcf.success_rate(yhat,y_test))


        row_dict['method'] = 'IBCF'
        row_dict['train'] = np.mean(tr_ibcf)
        row_dict['cv'] = np.mean(cv_ibcf)
        df_result = insert_result(df_result,row_dict)

        row_dict['method'] = 'UBCF'
        row_dict['train'] = np.mean(tr_ubcf)
        row_dict['cv'] = np.mean(cv_ubcf)
        df_result = insert_result(df_result,row_dict)


df_result = pd.DataFrame(df_result,columns=cols)
df_result.to_csv('recom_result.csv',index=False)
