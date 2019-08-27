#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 22:31:33 2019

@author: heimi
"""

import pandas as pd

#dd=pd.read_excel('../data/amazon_data/amazon_com_products 20190614.xlsx')

#dd.to_csv('../data/amazon_data/title_data.csv',index=False)
dd=pd.read_csv('../data/amazon_data/title_data.csv')

dd=dd[['keyword', 'product']].drop_duplicates()


#d_dict={}
#for v in dd.groupby('keyword'):
#    d_dict[v[0]]=v[1]
    












