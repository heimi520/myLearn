#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 16:42:32 2019

@author: heimi
"""

import pandas as pd

data_buy=pd.read_csv('../data/meorient_data/买家全行业映射标签（20190717）.csv')

data_buy=data_buy[(data_buy['T1']=='Apparel')|(data_buy['T1']=='Consumer Electronics')]

data_buy['source_text']=data_buy['PRODUCTS_NAME'].copy()
data_buy['fromLang']='auto'
data_buy['toLang']='en'


data_buy.to_csv('../data/pred_need_trans/parel_3c_all_need_trans.csv',index=False)





