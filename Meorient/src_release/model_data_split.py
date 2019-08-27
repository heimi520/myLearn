#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 13:44:20 2019

@author: heimi
"""


# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:05:29 2019

@author: Administrator
"""

#watch -n 10 nvidia-smi


import warnings
warnings.filterwarnings('ignore')
import sys
from os import path
sys.path.append( path.dirname(path.dirname(path.abspath(__file__))))


from model_config import *
from mylib.model_meorient import *


from DetectModel import *


import pandas as pd
from sklearn.model_selection import train_test_split

tag_args=TagModelArgs()
de_args=DetectModelArgs()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=tag_args.GPU_DEVICES # 使用编号为1，2号的GPU 
if tag_args.GPU_DEVICES=='-1':
    logger.info('using cpu')
else:
    logger.info('using gpu:%s'%tag_args.GPU_DEVICES)


import keras.backend.tensorflow_backend as KTF 
gpu_config=tf.ConfigProto()
gpu_config.gpu_options.per_process_gpu_memory_fraction=0.9 ##gpu memory up limit
gpu_config.gpu_options.allow_growth = True # 设置 GPU 按需增长,不易崩掉
session = tf.Session(config=gpu_config) 
# 设置session 
KTF.set_session(session)


print('data text clean....')
#
#data=pd.read_csv('../data/input/all_data_dp.csv')
#
#data=data[data['PRODUCT_NAME'].notnull()]
#data=data[data['PRODUCT_NAME'].apply(lambda x:len(x.replace(' ','').replace('\n',''))>=3)]
#
#"""dropduplicate"""
#data=data.groupby(['PRODUCT_NAME']).head(1).reset_index() 
#
#
#detectpipline=DetectPipLine(seq_len=de_args.SEQ_LEN,is_rename_tag=False,model_id=de_args.MODEL_ID)
#lang_pred=detectpipline.predict(data[['PRODUCT_NAME']])
#
#data['lang_pred']=lang_pred
#
#en_data=data[data['lang_pred']=='en']
#
#trans_data=pd.read_csv('../data/input/trans_sample_data.csv')
#
#cols_list=['lang_pred', 'PRODUCT_NAME', 'PRODUCT_NAME_TRANS', 'PRODUCT_TAG_NAME',
#       'PRODUCT_TAG_ID']
#en_data['PRODUCT_NAME_TRANS']=en_data['PRODUCT_NAME'].copy()
#en_data['lang_pred']='en'
#en_data=en_data[cols_list]
#
#dd=pd.concat([en_data,trans_data],axis=0)
#
#
#

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numpy.random import seed
import os
os.environ['PYTHONHASHSEED'] = '-1'
seed(0)  
rns=np.random.RandomState(0)

#dd=dd.rename(columns={'PRODUCT_NAME':'PRODUCT_NAME_ORIG','PRODUCT_NAME_TRANS':'PRODUCT_NAME'})
dd=pd.read_csv('../data/input/trans_sample_data.csv')


data_his,data_test = train_test_split(dd, test_size=0.05)

#data_his,data_test = dd[dd['BUYSELL']=='sell'],dd[dd['BUYSELL']=='buy']


#
#aa=data_his['PRODUCT_NAME'].apply(lambda x:len(x)).sort_values(ascending=False)
#aa.hist(bins=100)

data_his.to_csv('../data/input/data_his.csv',index=False)
data_test.to_csv('../data/input/data_test.csv',index=False)


























