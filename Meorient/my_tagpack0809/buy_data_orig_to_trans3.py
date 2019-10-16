#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:38:14 2019

@author: heimi
"""



import pandas as pd


data=pd.read_excel('../data/meorient_data/哈语翻译 .xlsx',encoding='gbk',sheetname=1)
data.columns=['name1','PRODUCTS_NAME','TAG_CODE']
data=data[data['PRODUCTS_NAME'].notnull()]

md=data[['PRODUCTS_NAME']].drop_duplicates()

md=md.rename(columns={'PRODUCTS_NAME':'source_text'})
md['fromLang']='auto'
md['toLang']='en'

md.to_csv('../data/tagpack/hayu_need_trans_en2.csv',index=False,encoding='utf8')






