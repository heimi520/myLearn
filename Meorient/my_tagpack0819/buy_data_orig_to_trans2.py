#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 09:32:53 2019

@author: heimi
"""



import pandas as pd


data=pd.read_excel('../data/meorient_data/买家全行业映射标签（20190717）.xlsx',encoding='gbk')

tag_stand=pd.read_excel('../data/tagpack2/8.19 打标标签.xlsx',sheetname=0)
tag_stand.columns=['PRODUCT_TAG_NAME','T1','T2']


data=pd.merge(data,tag_stand[['T1']].drop_duplicates(),on=['T1'],how='inner')


md=data[['PRODUCTS_NAME']].drop_duplicates()

md=md.rename(columns={'PRODUCTS_NAME':'source_text'})
md['fromLang']='auto'
md['toLang']='en'

md.to_csv('../data/tagpack2/tagpack2_buy_need_trans_T0.csv',index=False,encoding='utf8')













