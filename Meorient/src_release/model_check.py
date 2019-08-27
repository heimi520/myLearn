#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:50:46 2019

@author: heimi
"""

import pandas as  pd

data=pd.read_excel('../data/check/7.1部分检查：meorient_tag_pred_0628.xlsx')

pred=pd.read_csv('../data/output/meorient_tag_pred_rnn0701.csv')
#pred=pd.read_csv('../data/output/meorient_tag_pred_bigru0701.csv')

pred1=pd.read_csv('../data/output/meorient_tag_pred_rnn0701.csv')
pred2=pd.read_csv('../data/output/meorient_tag_pred_bigru0701.csv')

pred_all=pd.merge(pred1[['PRODUCT_NAME','TAG_NAME_PRED']],pred2[['PRODUCT_NAME','TAG_NAME_PRED']],on=['PRODUCT_NAME'],how='left')

aa_notmatch=pred_all[pred_all['TAG_NAME_PRED_x']!=pred_all['TAG_NAME_PRED_y']]
aa_same=pred_all[pred_all['TAG_NAME_PRED_x']==pred_all['TAG_NAME_PRED_y']]


dd=pd.merge(data,pred[['PRODUCT_NAME','TAG_NAME_PRED']],on=['PRODUCT_NAME'],how='left')


checkdata=dd[dd['是否检查']==1]

okdata=checkdata[checkdata['是否OK']==1]
restdata=checkdata[checkdata['是否OK']!=1]

bad2=okdata[okdata['TAG_NAME_PRED_x']!=okdata['TAG_NAME_PRED_y']]


rest_same=restdata[restdata['TAG_NAME_PRED_x']==restdata['TAG_NAME_PRED_y']]
rest_notmatch=restdata[restdata['TAG_NAME_PRED_x']!=restdata['TAG_NAME_PRED_y']]



#print('acc///',len(okdata)/len(checkdata))




















