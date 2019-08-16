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
from mylib.model_meorient_esemble import *
from detect_config import *

import pandas as pd
from sklearn.model_selection import train_test_split


rnnconfig=rnnConfig()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=rnnconfig.GPU_DEVICES # 使用编号为1，2号的GPU 
if rnnconfig.GPU_DEVICES=='-1':
    logger.info('using cpu')
else:
    logger.info('using gpu:%s'%rnnconfig.GPU_DEVICES)



def pipeline_predict(line_list):
    col=pd.DataFrame(line_list,columns=['PRODUCT_NAME'])  
    
    text2feature=Text2Feature(rnnconfig)

    x_test_padded_seqs=text2feature.pipeline_transform(col)
    
        
    rnnconfig.NUM_TAG_CLASSES=text2feature.num_classes_list[1]
    rnnconfig.TOKENIZER=text2feature.tokenizer 
    rnnconfig.CUT_LIST=text2feature.cut_list
     
    model=TextRNN(rnnconfig)
        
    model.build_model()
    
    [prob1,prob2]=model.predict(x_test_padded_seqs)

    tag1_max_int=np.argmax(prob1,axis=1)
    tag2_max_int=np.argmax(prob2,axis=1)

    [tag1_max,tag2_max]=text2feature.num2label([tag1_max_int, tag2_max_int])
    
    prob1_max=np.max(prob1,axis=1)
    prob2_max=np.max(prob2,axis=1)
    
    prob1_backup=prob1.copy()
    for k,v in enumerate(tag1_max_int):
        prob1_backup[k,v]=0
    prob2_backup=prob2.copy()
    for k,v in enumerate(tag2_max_int):
        prob2_backup[k,v]=0
    
    
    tag1_second_int=np.argmax(prob1_backup,axis=1)
    tag2_second_int=np.argmax(prob2_backup,axis=1)
    [tag1_second,tag2_second]=text2feature.num2label([tag1_second_int,tag2_second_int ])
    
    prob1_second=np.max(prob1_backup,axis=1)
    prob2_second=np.max(prob2_backup,axis=1)
    
    return tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second




data_test=pd.read_csv('../data/input/trans_sample_data.csv')

line_list=['huawei mate 20']
tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second=pipeline_predict(line_list)
print('tag_max',tag2_max,'T1 ',tag1_max)


line_list=data_test['PRODUCT_NAME'].tolist()
tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second=pipeline_predict(line_list)

data_test['T1_NAME_PRED1']=tag1_max
data_test['T1_NAME_PRED2']=tag1_second
data_test['T1_prob_pred1']=prob1_max
data_test['T1_prob_pred2']=prob1_second

data_test['TAG_NAME_PRED1']=tag2_max
data_test['TAG_NAME_PRED2']=tag2_second
data_test['TAG_prob_pred1']=prob2_max
data_test['TAG_prob_pred2']=prob2_second




data_test['TAG_NAME_PRED']=data_test['TAG_NAME_PRED1'].apply(lambda x:'Other' if len(re.findall('Other',x))>0 else x)
#data_test['TAG_NAME_PRED']=np.where(data_test['TAG_prob_pred1']>=0.4,data_test['TAG_NAME_PRED1'],'Other')

data_test2=data_test[['BUYSELL','TAG_NAME_PRED','TAG_NAME_PRED1','TAG_NAME_PRED2','TAG_prob_pred1','T1_NAME_PRED1','T1_prob_pred1','PRODUCT_NAME']]
data_test2.to_csv('../data/output/meorient_tag_pred_mlt_rnn.csv',index=False,encoding='utf-8')


cd=data_test.groupby('TAG_NAME_PRED')['TAG_NAME_PRED'].count().sort_values(ascending=False).to_frame('count')
cd['tag']=cd.index
cd['pct']=cd['count']/cd['count'].sum()
#cd.to_csv('../data/output/tag_weight.csv',index=False)


cd2=data_test.groupby('T1_NAME_PRED1')['T1_NAME_PRED1'].count().sort_values(ascending=False).to_frame('count')
cd2['tag']=cd2.index
cd2['pct']=cd2['count']/cd2['count'].sum()
#cd.to_csv('../data/output/tag_weight.csv',index=False)














