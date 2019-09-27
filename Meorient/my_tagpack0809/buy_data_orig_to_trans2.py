#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 09:32:53 2019

@author: heimi
"""



import pandas as pd

T_LEVEL='T0'

data=pd.read_excel('../data/meorient_data/买家无T1 TAG.xlsx',encoding='gbk')

tag_stand=pd.read_excel('../data/tagpack/8.8机器打标目标标签.xlsx',encoding='gbk')
tag_stand.columns=['PRODUCT_TAG_NAME','T1','T2']


if T_LEVEL=='T1':
    data=pd.merge(data,tag_stand[['T1']].drop_duplicates(),on=['T1'],how='inner')
elif T_LEVEL=='T2':
    data=pd.merge(data,tag_stand[['T1','T2']].drop_duplicates(),on=['T1','T2'],how='inner')
elif T_LEVEL=='T0':
    pass
    
md=data[['PRODUCT_NAME']].drop_duplicates()

md=md.rename(columns={'PRODUCT_NAME':'source_text'})
md['fromLang']='auto'
md['toLang']='en'
md=md[md['source_text'].notnull()]

md.to_csv('../data/tagpack/tagpack1_buy_need_trans_%s_new2.csv'%T_LEVEL,index=False,encoding='utf8')













