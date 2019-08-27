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


from model_config import *
from DetectModel import *


import pandas as pd
from sklearn.model_selection import train_test_split


de_args=DetectModelArgs()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=de_args.GPU_DEVICES # 使用编号为1，2号的GPU 
if de_args.GPU_DEVICES=='-1':
    logger.info('using cpu')
else:
    logger.info('using gpu:%s'%de_args.GPU_DEVICES)


import keras.backend.tensorflow_backend as KTF 
gpu_config=tf.ConfigProto()
gpu_config.gpu_options.per_process_gpu_memory_fraction=0.9 ##gpu memory up limit
gpu_config.gpu_options.allow_growth = True # 设置 GPU 按需增长,不易崩掉
session = tf.Session(config=gpu_config) 
# 设置session 
KTF.set_session(session)


data_his=pd.read_csv('../data/input/lang_his.csv')
data_test=pd.read_csv('../data/input/lang_test.csv')
data_test=pd.concat([data_his,data_test],axis=0)

cnt_pd=data_test.groupby('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME'].count().to_frame('count')

data_test=pd.read_csv('../data/input/lang_test.csv')

detectpipline=DetectPipLine(seq_len=de_args.SEQ_LEN,max_words=de_args.MAX_WORDS, voc_dim=de_args.VOC_DIM, \
                            is_rename_tag=False,model_id=de_args.MODEL_ID,is_pre_train=False,\
                      init_learn_rate=de_args.INIT_LEARN_RATE,batch_size=de_args.BATCH_SIZE,epoch_max=de_args.EPOCH_MAX,\
                      drop_out_rate=de_args.DROP_OUT_RATE,early_stop_count=de_args.EARLY_STOP_COUNT)

y_pred=detectpipline.predict(data_test)

data_test['TAG_NAME_PRED']=y_pred

data_test['sample_count']=data_test['PRODUCT_TAG_NAME'].map(cnt_pd['count'].to_dict())

bad=data_test[data_test['TAG_NAME_PRED']!=data_test['PRODUCT_TAG_NAME']]
bad=bad[['PRODUCT_NAME','PRODUCT_TAG_NAME','TAG_NAME_PRED','sample_count']]

#bad.to_csv('../data/output/lang_data_check.csv',index=False)

from sklearn import metrics

def ret_metrics(data_test):
    y_true_class,y_pred_class=data_test['label'],data_test['pred']
    acc=metrics.accuracy_score(y_true_class, y_pred_class)
    f1= metrics.f1_score(y_true_class, y_pred_class, average='weighted')
    print('准确率', acc)
    print('平均f1-score:', f1)
    

data_test['cnt']=1
data_test['ok']=(data_test['PRODUCT_TAG_NAME']==data_test['TAG_NAME_PRED']).astype(int)



#ret_pd=data_test.groupby(['PRODUCT_TAG_ID','label']).agg({'cnt':'sum','ok':'sum'}).reset_index() 
#ret_pd['acc']=(ret_pd['ok']/ret_pd['cnt']).round(3)
#



ret_pd=data_test.groupby('PRODUCT_TAG_NAME').agg({'ok':'sum', 'cnt': 'sum'})
ret_pd['acc']=ret_pd['ok']/ret_pd['cnt']
ret_pd['test_smaple_count']=data_test.groupby('PRODUCT_TAG_NAME')['ok'].count()
ret_pd=pd.merge(ret_pd,cnt_pd,left_index=True,right_index=True,how='left')
ret_pd=ret_pd.sort_values('acc')
ret_pd=ret_pd.rename(columns={'count':'train_sample_count'})



#buy_test=data_test[data_test['BUYSELL']=='buy']
#sell_test=data_test[data_test['BUYSELL']=='sell']
#
#print('buy metric')
#
#ret_metrics(buy_test)
#print('sell metric')
#ret_metrics(sell_test)


