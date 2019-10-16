#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 09:32:53 2019

@author: heimi
"""



import pandas as pd


data=pd.read_excel('../data/meorient_data/买家全行业映射标签（20190717）.xlsx',encoding='gbk')

tag_stand=pd.read_excel('../data/tagpack/8.8机器打标目标标签.xlsx',encoding='gbk')
tag_stand.columns=['PRODUCT_TAG_NAME','T1','T2']

data=pd.merge(data,tag_stand[['T1','T2']].drop_duplicates(),on=['T1','T2'],how='inner')



md=data[['PRODUCTS_NAME']].drop_duplicates()

md=md.rename(columns={'PRODUCTS_NAME':'source_text'})
md['fromLang']='auto'
md['toLang']='en'

md.to_csv('../data/tagpack/tagpack1_buy_need_trans.csv',index=False,encoding='utf8')












