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
from sklearn.model_selection import train_test_split


rnnconfig=rnnConfig()
cnnconfig=cnnConfig()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=rnnconfig.GPU_DEVICES # 使用编号为1，2号的GPU 
if rnnconfig.GPU_DEVICES=='-1':
    logger.info('using cpu')
else:
    logger.info('using gpu:%s'%rnnconfig.GPU_DEVICES)


data_test=pd.read_csv('../data/input/trans_sample_data.csv')
#data_test=data_test.sample(1000)
#
#line_list=['huawei mate 20']
#tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second=pipeline_predict_esemble(line_list)
#print('tag_max',tag2_max,'T1 ',tag1_max)


line_list=data_test['PRODUCT_NAME'].tolist()
#tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second=pipeline_predict_esemble(line_list,['rnn','cnn'])

tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second=pipeline_predict_esemble(line_list,['cnn'])
data_test['T1_NAME_PRED1']=tag1_max
data_test['T1_NAME_PRED2']=tag1_second
data_test['T1_prob_pred1']=prob1_max
data_test['T1_prob_pred2']=prob1_second

data_test['TAG_NAME_PRED1']=tag2_max
data_test['TAG_NAME_PRED2']=tag2_second
data_test['TAG_prob_pred1']=prob2_max
data_test['TAG_prob_pred2']=prob2_second

tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second=pipeline_predict_esemble(line_list,['rnn'])
data_test['T1_NAME_PRED1rnn']=tag1_max
data_test['T1_NAME_PRED2rnn']=tag1_second
data_test['T1_prob_pred1rnn']=prob1_max
data_test['T1_prob_pred2rnn']=prob1_second

data_test['TAG_NAME_PRED1rnn']=tag2_max
data_test['TAG_NAME_PRED2rnn']=tag2_second
data_test['TAG_prob_pred1rnn']=prob2_max
data_test['TAG_prob_pred2rnn']=prob2_second



tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second=pipeline_predict_esemble(line_list,['rnn','cnn'])
data_test['T1_NAME_PRED1all']=tag1_max
data_test['T1_NAME_PRED2all']=tag1_second
data_test['T1_prob_pred1all']=prob1_max
data_test['T1_prob_pred2all']=prob1_second

data_test['TAG_NAME_PRED1all']=tag2_max
data_test['TAG_NAME_PRED2all']=tag2_second
data_test['TAG_prob_pred1all']=prob2_max
data_test['TAG_prob_pred2all']=prob2_second


data_test['TAG_NAME_PRED']=data_test['TAG_NAME_PRED1'].apply(lambda x:'Other' if len(re.findall('Other',x))>0 else x)
#data_test['TAG_NAME_PRED']=np.where(data_test['TAG_prob_pred1']>=0.4,data_test['TAG_NAME_PRED1'],'Other')

data_test2=data_test[['BUYSELL','TAG_NAME_PRED','TAG_NAME_PRED1','TAG_NAME_PRED2','TAG_prob_pred1','T1_NAME_PRED1','T1_prob_pred1','PRODUCT_NAME']]
data_test2.to_csv('../data/output/meorient_tag_pred_mlt_esemble.csv',index=False,encoding='utf-8')


cd=data_test.groupby('TAG_NAME_PRED')['TAG_NAME_PRED'].count().sort_values(ascending=False).to_frame('count')
cd['tag']=cd.index
cd['pct']=cd['count']/cd['count'].sum()
#cd.to_csv('../data/output/tag_weight.csv',index=False)


cd2=data_test.groupby('T1_NAME_PRED1')['T1_NAME_PRED1'].count().sort_values(ascending=False).to_frame('count')
cd2['tag']=cd2.index
cd2['pct']=cd2['count']/cd2['count'].sum()
#cd.to_csv('../data/output/tag_weight.csv',index=False)


aa=data_test[data_test['TAG_NAME_PRED1']!=data_test['TAG_NAME_PRED1rnn']]
aa=aa[['TAG_NAME_PRED1','TAG_NAME_PRED1rnn','TAG_NAME_PRED1all','T1_NAME_PRED1','T1_NAME_PRED1rnn','PRODUCT_NAME']]










