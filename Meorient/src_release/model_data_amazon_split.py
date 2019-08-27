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
dd=pd.read_csv('../data/amazon_data/title_data.csv')

dd=dd[['keyword', 'product']].drop_duplicates()
dd=dd.rename(columns={'keyword':'PRODUCT_TAG_NAME','product':'PRODUCT_NAME'})
dd['PRODUCT_TAG_ID']=dd['PRODUCT_TAG_NAME'].copy()
dd['BUYSELL']='sell'

#
#tag_dict=pd.read_csv('../data/input/tag_dict.csv')
#
#dd['PRODUCT_TAG_NAME2']=dd['PRODUCT_TAG_NAME'].copy()
#
#
#base_dict=dd.set_index('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME2'].to_dict()
#tag2new_dict={'PC Adapters'	:'Adapters',
#'Camera Video Glasses':	'Video Glasses',
#'Women Downcoats & Jackets':'Women Downcoats',
#'Women Cotton Padded Coats & Jackets':'Women Cotton Padded Coats',
#'Women Leather Coats & Jackets':'Women Leather Coats',
#'Men Downcoats & Jackets': 	'Men Downcoats',
#'Men Leather Coats & Jackets':'Men Leather Coats',
#'Men Cotton Padded Coats & Jackets': 	'Men Cotton Padded Coats',
#'Women Spring Coats':'Women Coats',
#'Men Spring Coats':	'Men Coats',
#}
#
#tag2new_dict={v:k for k,v in tag2new_dict.items()}
#base_dict.update(tag2new_dict)
#
#
#dd['PRODUCT_TAG_NAME']=dd['PRODUCT_TAG_NAME'].map(base_dict)
#
##aa=dd.loc[dd['PRODUCT_TAG_NAME']!=dd['PRODUCT_TAG_NAME2'],['PRODUCT_TAG_NAME','PRODUCT_TAG_NAME2']].drop_duplicates()
##
#
#
#tag_amazon=set(dd['PRODUCT_TAG_NAME'])
#tag_mi=set(tag_dict['PRODUCT_TAG_NAME'])
#tag_trans=set(tag2new_dict.values())
#
#tag_select=tag_mi-tag_trans
#
##tag_sect=tag_amazon.intersection(tag_mi)
#
#tag_pd=pd.DataFrame(tag_select,columns=['PRODUCT_TAG_NAME'])
#
#okdata=pd.merge(dd[['PRODUCT_TAG_NAME','PRODUCT_NAME']],tag_pd,on=['PRODUCT_TAG_NAME'],how='inner')
#okdata['PRODUCT_TAG_ID']=okdata['PRODUCT_TAG_NAME'].copy()
#
#


data_his,data_test = train_test_split(dd, test_size=0.05)


data_his.to_csv('../data/input/amazon_his.csv',index=False)
data_test.to_csv('../data/input/amazon_test.csv',index=False)


























