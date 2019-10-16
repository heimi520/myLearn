#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:59:07 2019

@author: heimi
"""


import pandas as pd
import numpy as np

data=pd.read_excel('../data/meorient_data/哈语翻译 .xlsx',encoding='gbk',sheetname=0)
data.columns=['T1_NAME','PRODUCTS_NAME']
data=data[data['PRODUCTS_NAME'].notnull()]
md=data[['PRODUCTS_NAME']].drop_duplicates()


dd1=pd.read_csv('../data/tagpack/hayu_trans_zh.csv')
dd2=pd.read_csv('../data/tagpack/hayu_trans.csv')



dd=pd.merge(dd1[['source_text','trans_text']],dd2[['source_text','trans_text']],on=['source_text'],how='left')
dd.columns=['PRODUCTS_NAME','zh','en']

ret=pd.merge(data,dd,on=['PRODUCTS_NAME'],how='left')


#ret=pd.DataFrame(ret,columns=['PRODUCTS_NAME','en','zh'])
#
#
#aa=pd.merge(data,ret,on=['PRODUCTS_NAME'],how='left')
ret.to_excel('../data/tagpack/trans_T1_new.xlsx',encoding='gbk',index=False)
##





