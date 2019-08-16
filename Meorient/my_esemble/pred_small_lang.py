#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 23:31:12 2019

@author: heimi
"""


import warnings
warnings.filterwarnings('ignore')
import sys
from os import path
sys.path.append( path.dirname(path.dirname(path.abspath(__file__))))


from mytools import *

import pandas as pd


data,data_test=read_orig_data()
tag_max,tag_second,prob_max,prob_second =detect_pipeline_predict(data_test)
data_test['lang_pred1']=tag_max
data_test['lang_pred2']=tag_max
data_test['prob_max']=prob_max
data_test['prob_second']=prob_second


cols=['PRODUCT_NAME_ORIG','lang_pred1']
small_data=data_test.loc[data_test['lang_pred1']!='en',cols]
small_data=small_data.rename(columns={'lang_pred1':'fromLang'})
small_data['toLang']='en'
small_data=small_data.rename(columns={'PRODUCT_NAME_ORIG':'source_text'})
small_data.to_csv('../data/pred_need_trans/small_data.csv',index=False)

en_data=data_test.loc[data_test['lang_pred1']=='en',cols]
en_data=en_data.rename(columns={'PRODUCT_NAME_ORIG':'source_text'})
en_data['trans_text']=en_data['source_text'].copy()
en_data=en_data.rename(columns={'lang_pred1':'fromLang'})
en_data['toLang']='en'
en_data['batch_idx']=0
en_data['idx']=0

en_data.to_csv('../data/pred_need_trans/en_data.csv',index=False)





