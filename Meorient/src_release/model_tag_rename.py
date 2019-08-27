#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 16:19:35 2019

@author: heimi
"""



import pandas as pd



data_his=pd.read_csv('../data/input/ali_amazon_his.csv')
data_test=pd.read_csv('../data/input/ali_amazon_test.csv')


dd=pd.concat([data_his,data_test],axis=0)



md=dd.groupby('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME'].count().sort_values().to_frame('count')

md.to_csv('../data/output/tag_rename.csv')



