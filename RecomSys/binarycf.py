#!/usr/bin/python3 -tt
'''------------------------------------- >Last Modified on Sun, 29 Jan 2017< '''
'''
- df_long (customer, category, productid)
- df_common (customer, cat1, cat2)
- X df_common.cat1 series of sets of items (feature) 
- y df_common.cat2 series of sets of items (target)
- metric similarity metric

'''

from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import pandas as pd
from sparsesvd import sparsesvd as svd

def chk():
    return('test')

############################################################# similarity metric
def common_similarity(set1,set2):
    return(len(set1.intersection(set2)))

# cosine-similarity 
def cosine_similarity(set1,set2):
    iprod = len(set1.intersection(set2))
    a_len, b_len = np.sqrt(len(set1)), np.sqrt(len(set2))
    return(iprod/(a_len*b_len))

# normalisation (correlation)
def correlation(set1,set2,size):
    a_bar, b_bar = len(set1)/size, len(set2)/size
    l1, l2 = np.sqrt(size*a_bar*(1-a_bar)), np.sqrt(size*b_bar*(1-b_bar))
    iprod = len(set1.intersection(set2))
    return((iprod-size*a_bar*b_bar)/(l1*l2))

def correlation_similarity(size):
    def sim(set1,set2):
        return(correlation(set1,set2,size))
    return(sim)

###################################################################

def customer_in_cat(df_long,cat):
    '''
    Returns the data frame with columns customer and cat
    Each row describes a customer 

    - cat : category (str)

    - customer : customerid (num)
    - cat : set of items in the given category
    '''
    df_new = df_long[df_long.category==cat].copy()
    df_new = df_new.groupby('customer')['item'].apply(lambda x: set(x))
    df_new = df_new.reset_index()
    df_new.rename(columns={'item':cat},inplace=True)
    return(df_new)




def success_rate(yhat,ytrue):
    '''
    Compute the probability that there is an item in recommendation (yhat)
    which will be bought (ytrue).

    yhat  : series of sets (recommendation)
    ytrue : series of sets (purchased items)
    '''
    df = pd.concat([yhat,ytrue],axis=1)
    success = df.apply(lambda x: len(set.intersection(x[0],x[1]))>0, axis=1)
    return(success.mean())

    #success = [len(set.intersection(a,b))>0 for a,b in zip(yhat,ytrue)]
    #return(np.mean(success))




####################################################################

def get_popular_items(y,num=5):
    '''
    returns the set of most popular items
    '''
    rank = defaultdict(int)
    for items in y:
        for item in items:
            rank[item] += 1
    return(set(sorted(rank.keys(),key=lambda item:rank[item], reverse=True)[0:num]))


def ubcf(customer,X,y,metric=common_similarity,k=10,num=5):
    '''
    User-based collaborative filtering
    Returns a set of recommended items

    k      : the number of neighbours (similar customers)
    metric : similarity metric (function taking two sets) 
    num    : the number of recommendations
    '''

    sim_score = X.apply(lambda x: metric(x,customer)) ## series of scores
    ## pick the top-k neighbours (list of locations (not the index))
    sim_customers = list(sim_score.sort_values(ascending=False).index[range(k)])

    item_scores = defaultdict(float) ## dict: items -> score
    for sim_customer in sim_customers:
        for item in y.ix[sim_customer]: ## a set of items of a similar customer
            item_scores[item] += sim_score[sim_customer]

    top_items = sorted(item_scores.keys(), 
                       key=lambda x:item_scores[x], 
                       reverse=True)[0:num]
    return(set(top_items))


def compute_similar_items(df_common,metric=common_similarity,k=20):
    '''
    Compute a set of similar items for each item
    Returns a dict (productid (feature) -> set of productid (target))
    
    df_common : data frame (customer, cat1, cat2)
    metric    : function : (set1,set2) -> similarity 
    k       : number of similar items
    '''

    who_buys_X = defaultdict(set) ## item (feature) -> set of customerid
    who_buys_y = defaultdict(set) ## item (target) -> set of customerid

    for i in range(df_common.shape[0]):
        row = df_common.iloc[i]
        customer = row[0]
        for item in row[1]: ## feature
            who_buys_X[item].add(customer)
        for item in row[2]: ## target
            who_buys_y[item].add(customer)

    similar_items = dict() ## item (feature) -> (item (target) -> score)
    for item1 in who_buys_X.keys():
        scores = dict() ## item (target) -> score
        for item2 in who_buys_y.keys():
            scores[item2] = metric(who_buys_X[item1],who_buys_y[item2])

        top_items = sorted(who_buys_y.keys(), key=lambda y: scores[y], 
                           reverse=True)[0:k]

        similar_items[item1] = dict()
        for item2 in top_items:
            similar_items[item1][item2] = scores[item2]

    return(similar_items)



def ibcf(items,similar_items,num=5):
    '''
    Item-based collaborative filtering 
    Returns a set of items 
    
    items         : set of items 
    similar_items : precalculated dict (use compute_similar_items())
    num           : number of recommendations
    '''

    recom_score = defaultdict(float)
    for item in items:
        if item in similar_items.keys():
            for i,s in similar_items[item].items(): ## similar item -> score
                recom_score[i] += s
    top_items = sorted(recom_score.keys(),
                       key=lambda x: recom_score[x],
                       reverse=True)[0:num]
    return(set(top_items))


########################### SVD

def sets_to_matrix(series):
    '''
    Create a sparse matrix from a series of items.
    Returns the sparse matrix and a list of items.
    (We need the latter to get the item corresponding to an index.)
    '''

    item_list = set() ## list of items
    for items in series:
        item_list = item_list.union(items)
    item_list = sorted(list(item_list))

    item_to_index = dict() ## item -> its index
    for j in range(len(item_list)):
        item_to_index[item_list[j]] = j

    rows = []
    cols = []
    data = []
    for i in np.arange(len(series)):
        items = series[i]
        for item in items:
            rows.append(i)
            cols.append(item_to_index[item])
            data.append(1)

    X = sp.csc_matrix((data,(rows,cols)))
    return(X,item_list)



def matrix_decomposition(X,k=5):
    '''
    Decompose a given *sparse* matrix X into A and Bt such that X = A Bt.
    Here A is X.shape[0] x k and Bt is k x X.shape[1].

    This function uses SVD:
        X = U S Vt = (U sqrt(S)) (sqrt(S) Vt)
                     ^^^^^^^^^^^ ^^^^^^^^^^^^
                          A           Bt
    Returns A, Bt and the right inverse of Bt.
    Since Vt V is the identity matrix of size k, 
        (sqrt(S) Vt) (V sqrt(Sinv)) = sqrt(S) (Vt V) sqrt(S)^{-1} = 1
    '''

    Ut,s,Vt = svd(X,k)
    rs = np.diag(np.sqrt(s))  ## sqrt(S)

    A = np.dot(np.transpose(Ut),rs)
    Bt = np.dot(rs,Vt)

    rsinv = np.diag(np.sqrt(1/s)) ## sqrt(S^{-1})
    Brinv = np.dot(np.transpose(Vt),rsinv) 

    return(A,Bt,Brinv)
