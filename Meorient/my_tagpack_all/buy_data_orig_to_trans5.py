#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 09:32:53 2019

@author: heimi
"""



import pandas as pd


data=pd.read_excel('../data/trans_data/翻译 阿拉伯语，俄语.xlsx',encoding='gbk')
data.columns=['PRODUCT_NAME']

#tag_stand=pd.read_excel('../data/tagpack2/8.19 打标标签.xlsx',sheetname=0)
#tag_stand.columns=['PRODUCT_TAG_NAME','T1','T2']
#

#data=pd.merge(data,tag_stand[['T1']].drop_duplicates(),on=['T1'],how='inner')


md=data[['PRODUCT_NAME']].drop_duplicates()

md=md.rename(columns={'PRODUCT_NAME':'source_text'})
md['fromLang']='auto'
md['toLang']='ru'

md.to_csv('../data/trans_data/翻译 阿拉伯语，俄语_need_ru_transed.csv',index=False,encoding='utf8')













