#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 10:38:47 2019

@author: heimi
"""


import pandas as pd

cols=['PRODUCT_TAG_NAME','tag1','tag2','tag3','except']

reg_all=pd.read_csv('../data/reg_all/reg_my_v2.csv')


reg1=pd.read_csv('../data/reg_all/reg_apparal_3c.csv')
reg1.columns=cols

reg2=pd.read_csv('../data/reg_all/reg_0808.csv')
reg2=reg2.iloc[:,2:7]
reg2.columns=cols

reg3=pd.read_csv('../data/reg_all/reg_0819.csv')
reg3=reg3.iloc[:,[0,4,5,6,7]]
reg3.columns=cols

reg4=pd.read_csv('../data/reg_all/reg_0830.csv')
reg4=reg4.iloc[:,1:]
reg4.columns=cols


reg5=pd.read_csv('../data/reg_all/reg_0924.csv')
reg5.columns=cols


ret=pd.concat([reg1,reg2,reg3,reg4,reg5 ],axis=0)


md=reg_all[['PRODUCT_TAG_NAME']]
md=pd.merge(md,ret,on=['PRODUCT_TAG_NAME'],how='left')
idx=md['tag1'].isnull()
#
md.loc[idx,:]=reg_all.loc[idx,:]
md=md.fillna('')
md.to_csv('../data/tagpack_all/reg_merge.csv',index=False)















