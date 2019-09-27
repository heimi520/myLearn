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

from cnn_config import *
from mylib.model_meorient_esemble import *
#from detect_config import *

import pandas as pd
from sklearn.model_selection import train_test_split


cnnconfig=cnnConfig_STEP1()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=cnnconfig.GPU_DEVICES # 使用编号为1，2号的GPU 
if cnnconfig.GPU_DEVICES=='-1':
    logger.info('using cpu')
else:
    logger.info('using gpu:%s'%cnnconfig.GPU_DEVICES)



def pipeline_predict(line_list):
    col=pd.DataFrame(line_list,columns=['PRODUCT_NAME'])  
    
    text2feature=Text2Feature(cnnconfig)

    x_test_padded_seqs=text2feature.pipeline_transform(col)
    
        
    cnnconfig.NUM_TAG_CLASSES=text2feature.num_classes_list[1]
    cnnconfig.TOKENIZER=text2feature.tokenizer 
    cnnconfig.CUT_LIST=text2feature.cut_list
     
    model=TextCNN(cnnconfig)
        
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




def read_data():
    ok_other=pd.read_csv('../data/input/tagpack_ok_other.csv')
    data=ok_other[ok_other['sample_w']==1]
    data_other=ok_other[ok_other['sample_w']!=1]
    cnt_pd=data.groupby('T2')['T2'].count().to_frame('count')
    return data,data_other,cnt_pd



data,data_other,cnt_pd=read_data()
cols_list=['PRODUCT_NAME','T2','T1','PRODUCT_TAG_NAME','sample_w', 'source']
data=data[cols_list]

data_test=data

line_list=data_test['PRODUCT_NAME'].tolist()
tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second=pipeline_predict(line_list)

data_test['T1_NAME_PRED']=tag1_max
data_test['TAG_NAME_PRED']=tag2_max

data_test['cnt']=1
data_test['ok']=(data_test['PRODUCT_TAG_NAME']==data_test['TAG_NAME_PRED']).astype(int)


ret_copy=data_test.copy()

idx_bad=data_test['ok']==0
#
data_test.loc[idx_bad,'T1']=data_test.loc[idx_bad,'T1_NAME_PRED'].copy()
data_test.loc[idx_bad,'PRODUCT_TAG_NAME']=data_test.loc[idx_bad,'TAG_NAME_PRED'].copy()
data_test.loc[idx_bad,'sample_w']=0.9
data_test.loc[idx_bad,'source']='step1_pred'


#aa=data_test[idx_bad]
#
#data_test.groupby('sample_w')['T1'].count()
#

cols=list(data_other.columns)
cols.remove('tag_orig')

data_test=data_test[cols]

data_step1=pd.concat([data_test,data_other],axis=0)

data_step1.groupby('source')['T1'].count()


aa=data_test[idx_bad]

data_step1.to_csv('../data/input/tagpack_ok_other_step1.csv',index=False)





















