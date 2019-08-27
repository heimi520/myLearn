# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 17:38:15 2019

@author: Administrator
"""

import pandas as pd

dd1=pd.read_csv('../data/output/6.11 打标样本测试_result_check_20190611.csv')
dd2=pd.read_csv('../data/output/6.11 打标样本测试_result_check_20190612.csv')

dd=pd.concat([dd1,dd2],axis=1)

dd=pd.merge(dd1,dd2,left_index=True,right_index=True)

ret=dd[['PRODUCT_NAME_ORIG','PRODUCT_NAME_y','TAG_NAME_PRED_x','TAG_NAME_PRED_y']]

















