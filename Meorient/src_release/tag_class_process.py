# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 15:59:09 2019

@author: Administrator
"""


import pandas as pd
#
#dd1=pd.read_excel('../data/tag_class/3C-ProductTag0617.xlsx')
#dd2=pd.read_excel('../data/tag_class/服装-ProductTAG0617.xlsx')
#
#dd=pd.concat([dd1,dd2],axis=0)
#cols=['Product Tag', 'Product Tag ID']
#dd=dd[cols]
#dd=dd.rename(columns={'Product Tag':'PRODUCT_TAG_NAME', 'Product Tag ID':'PRODUCT_TAG_ID'})
#dd.to_csv('../data/input/tag_dict.csv',index=False)



dd1=pd.read_excel('../data/tag_class/服装&3C映射表v2.xlsx',sheet_name=0)
dd2=pd.read_excel('../data/tag_class/服装&3C映射表v2.xlsx',sheet_name=1)
dd=pd.concat([dd1,dd2],axis=0)
cols=['Product Tag', 'Product Tag ID']
dd=dd[cols]
dd=dd.rename(columns={'Product Tag':'PRODUCT_TAG_NAME', 'Product Tag ID':'PRODUCT_TAG_ID'})
dd.to_csv('../data/input/tag_dict.csv',index=False)





















