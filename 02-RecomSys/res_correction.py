#!/usr/bin/python3 -tt
'''------------------------------------- >Last Modified on Sun, 29 Jan 2017< '''
'''

'''

#from collections import defaultdict # for non-existing key of dict
#from collections import Counter # count the elements of a list

import numpy as np
#import scipy.stats as stats
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#import re
#import json # dict = json.load(json) # json.dump(data,file_obj)

#from sklearn.grid_search import GridSearchCV

df = pd.read_csv('recom_result.csv')
print(df.shape)

df_with = df[df.method == 'popular']
df_without = df[df.method != 'popular']

gb = df_with.groupby(['cat1','cat2','method'])
df_with = gb.aggregate(np.mean).reset_index()

dg = pd.concat([df_with,df_without],axis=0)
print(dg.shape)
print(8*7*4+dg.shape[0])

dg.to_csv('hoge.csv',index=False)
