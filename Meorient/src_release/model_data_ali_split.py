#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 13:44:20 2019

@author: heimi
"""

#watch -n 10 nvidia-smi


import warnings
warnings.filterwarnings('ignore')
import sys
from os import path
sys.path.append( path.dirname(path.dirname(path.abspath(__file__))))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numpy.random import seed
import os
os.environ['PYTHONHASHSEED'] = '-1'
seed(0)  
rns=np.random.RandomState(0)

#dd=dd.rename(columns={'PRODUCT_NAME':'PRODUCT_NAME_ORIG','PRODUCT_NAME_TRANS':'PRODUCT_NAME'})
dd=pd.read_csv('../data/ali_data/alibaba_products.csv')
#['kw', 'mainProducts', '序号', 'productName']

dd=dd[['kw', 'mainProducts']].drop_duplicates()
dd=dd.rename(columns={'kw':'PRODUCT_TAG_NAME','mainProducts':'PRODUCT_NAME'})
dd['PRODUCT_TAG_ID']=dd['PRODUCT_TAG_NAME'].copy()
dd['BUYSELL']='sell'

dd=dd[dd['PRODUCT_NAME'].notnull()]

data_his,data_test = train_test_split(dd, test_size=0.05)


data_his.to_csv('../data/input/ali_his.csv',index=False)
data_test.to_csv('../data/input/ali_test.csv',index=False)


























