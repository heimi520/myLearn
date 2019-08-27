# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 18:21:02 2019

@author: Administrator
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numpy.random import seed
import os
os.environ['PYTHONHASHSEED'] = '-1'
seed(0)  
rns=np.random.RandomState(0)

data=pd.read_csv('../data/input_lang/lang_data_merge.csv')
data=data.rename(columns={'trans_text':'PRODUCT_NAME','langto':'PRODUCT_TAG_ID'})
data['PRODUCT_TAG_NAME']=data['PRODUCT_TAG_ID'].copy()

data=data[data['PRODUCT_NAME'].notnull()]

data_his,data_test = train_test_split(data, test_size=0.1)

data_his.to_csv('../data/input/lang_his.csv',index=False)
data_test.to_csv('../data/input/lang_test.csv',index=False)








