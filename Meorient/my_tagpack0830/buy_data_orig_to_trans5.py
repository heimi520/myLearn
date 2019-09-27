#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 09:32:53 2019

@author: heimi
"""



import pandas as pd


data=pd.read_csv('../data/meorient_data/预注册买家提交标签信息表9.27.csv')


md=data[['PRODUCT_NAME']].drop_duplicates()

md=md.rename(columns={'PRODUCT_NAME':'source_text'})
md['fromLang']='auto'
md['toLang']='en'

md.to_csv('../data/meorient_data/预注册买家提交标签信息表9.27_need_transed.csv',index=False,encoding='utf8')













