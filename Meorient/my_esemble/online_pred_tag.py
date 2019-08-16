#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 23:31:12 2019

@author: heimi
"""

import warnings
warnings.filterwarnings('ignore')
import sys
from os import path
sys.path.append( path.dirname(path.dirname(path.abspath(__file__))))

from  my_textrnn.rnn_config import *
from  my_textcnn.cnn_config import *
from mytools import *
from detect_config import *
import pandas as pd


data,coldata=read_orig_data()
data_test=read_trans_data()

line_list=data_test['PRODUCT_NAME'].tolist()
tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second=pipeline_predict_esemble(line_list,['cnn','rnn'])

data_test['T1_NAME_PRED1']=tag1_max
data_test['T1_NAME_PRED2']=tag1_second
data_test['T1_prob_pred1']=prob1_max
data_test['T1_prob_pred2']=prob1_second

data_test['TAG_NAME_PRED1']=tag2_max
data_test['TAG_NAME_PRED2']=tag2_second
data_test['TAG_prob_pred1']=prob2_max
data_test['TAG_prob_pred2']=prob2_second

data_test.to_csv('../data/output/tag_pred.csv',index=False)

data_test['TAG_NAME_PRED']=data_test['TAG_NAME_PRED1'].apply(lambda x:'Other' if len(re.findall('Other',x))>0 else x)

data_test2=data_test[['TAG_NAME_PRED','TAG_NAME_PRED1','TAG_NAME_PRED2','TAG_prob_pred1','T1_NAME_PRED1','T1_prob_pred1','PRODUCT_NAME']]
data_test2['key']=data_test['TAG_NAME_PRED1'].apply(lambda x:re.sub('\s+','',x.lower()))

data_test2.to_csv('../data/output/meorient_tag_pred_enlish.csv',index=False,encoding='utf-8')


cd=data_test.groupby('TAG_NAME_PRED')['TAG_NAME_PRED'].count().sort_values(ascending=False).to_frame('count')
cd['tag']=cd.index
cd['pct']=cd['count']/cd['count'].sum()
#cd.to_csv('../data/output/tag_weight.csv',index=False)

cd2=data_test.groupby('T1_NAME_PRED1')['T1_NAME_PRED1'].count().sort_values(ascending=False).to_frame('count')
cd2['tag']=cd2.index
cd2['pct']=cd2['count']/cd2['count'].sum()
#cd.to_csv('../data/output/tag_weight.csv',index=False)


data_ret=pd.merge(data,data_test[['TAG_NAME_PRED','TAG_NAME_PRED1','T1_NAME_PRED1','PRODUCT_NAME','PRODUCT_NAME_ORIG','fromLang']],on=['PRODUCT_NAME_ORIG'],how='left')
data_ret['key']=data_ret['TAG_NAME_PRED1'].apply(lambda x:re.sub('\s+','',str(x).lower()))

#tag_select=pd.read_excel('../data/meorient_data/服装3C 标签正确率80%+.xlsx')
tag_select=pd.read_excel('../data/meorient_data/服装3c第二批验收需求.xlsx')
tag_select.columns=['PRODUCT_TAG_NAME','PRODUCT_TAG_ID_NEW']
#tag_select.columns=['PRODUCT_TAG_NAME','PRODUCT_TAG_ID_NEW','acc_his']
tag_select['key']=tag_select['PRODUCT_TAG_NAME'].apply(lambda x:re.sub('\s+','',x.lower()))

data_show=pd.merge(data_ret,tag_select[['PRODUCT_TAG_ID_NEW','key']],on=['key'],how='inner')


bb=pd.merge(data_test2,tag_select[['PRODUCT_TAG_ID_NEW','key']],on=['key'],how='inner')
bb.to_excel('../data/output/hot_tag_need_to_check_apparel_3c.xlsx',index=False,encoding='gbk')


#data_show.to_csv('../data/output/meorient_buy_sell_data_tag.csv',index=False,encoding='gbk')
#data_show.to_excel('../data/output/meorient_buy_sell_data_tag.xlsx',index=False,encoding='gbk')
#data_show.to_excel('../data/output/meorient_buy_sell_data_tag_HOT.xlsx',index=False,encoding='gbk')


aa=data_show.groupby('key')['key'].count().to_frame('count')




#cols_list=['BUYSELL', 
#           'PRODUCT_TAG_NAME', 'TAG_NAME_PRED1',
#           'T1_NAME_PRED1','PURCHASER_ID', 'SOURCE_TYPE','COUNTRY_ID', 'PRODUCT_TAG_ID','PRODUCT_TAG_ID_NEW','SUPPLIER_ID', 'T1',
#           'T1_ID', 'TAG_NAME_PRED', 'fromLang','PRODUCT_NAME']



#
#data_show=data_show[cols_list]
#data_show=data_show.astype(str)
#output_name='output/%s.xlsx'%(tag_name.replace('/',''))
#writer = pd.ExcelWriter('../data/output/data_meorient_select_ok.csv')

#data_show.to_excel('../data/output/data_meorient_select_ok222222222.xlsx',index=False)


