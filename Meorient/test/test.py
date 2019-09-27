#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:16:02 2019

@author: heimi
"""

import pandas as pd

#dd=pd.read_excel('../data/meorient_data/demo_result.xlsx')
dd=pd.read_csv('../data/meorient_data/demo_result.csv')
dd.columns=['PRODUCT_NAME','TAG_NAME_PRED','TAG_dest','tag_real','check']

dd['check']=dd['check'].astype(int)

#
#dd=dd.groupby('tag_real').head(10)
#dd.to_csv('../data/meorient_data/demo_result.csv',index=False)

ret=dd.groupby('tag_real')[['check']].count()
ret.columns=['num_real']
ret['num_pred_true']=dd.groupby('tag_real')['check'].sum()  
ret['acc']=ret['num_pred_true']/ret['num_real']
ret.columns=['实际数量','正确数量 ','单标签准确率']



