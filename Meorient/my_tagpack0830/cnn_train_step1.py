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

from cnn_config import *
from mylib.model_meorient_esemble import *

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

cnnconfig=cnnConfig_STEP1()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=cnnconfig.GPU_DEVICES # 使用编号为1，2号的GPU 
if cnnconfig.GPU_DEVICES=='-1':
    logger.info('using cpu')
else:
    logger.info('using gpu:%s'%cnnconfig.GPU_DEVICES)



import keras.backend.tensorflow_backend as KTF 
gpu_config=tf.ConfigProto()
gpu_config.gpu_options.per_process_gpu_memory_fraction=0.95 ##gpu memory up limit
gpu_config.gpu_options.allow_growth = True # 设置 GPU 按需增长,不易崩掉
session = tf.Session(config=gpu_config) 
# 设置session 
KTF.set_session(session)

import keras 
#keras.backend.set_floatx('float16')

logger.info('keras floatx:%s'%(keras.backend.floatx()))

import multiprocessing
from multiprocessing import *

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import random


data_add=pd.read_csv('../data/tagpack0830/data_filter_added_step1.csv')

data_other=pd.read_csv('../data/tagpack0830/other_step1.csv')
data_other['sample_w']=0.95


data=pd.concat([data_add,data_other],axis=0)

 
random.seed(1)

cols_list=[ 'PRODUCT_NAME','PRODUCT_TAG_NAME','T1','sample_w', 'source']
total_data=data[cols_list]

total_data.info()

total_data['source'].unique()


data_train,data_val= train_test_split(total_data,test_size=0.05)
logger.info('total_data shape//////%s'%total_data.shape[0])
text2feature=Text2Feature(cnnconfig)
x_train_padded_seqs,x_val_padded_seqs,y1_train,y1_val,y2_train,y2_val,w_sp_train=text2feature.pipeline_fit_transform(data_train,data_val)

cnnconfig.NUM_TAG_CLASSES=text2feature.num_classes_list[1]
cnnconfig.TOKENIZER=text2feature.tokenizer 
cnnconfig.CUT_LIST=text2feature.cut_list
 
model=TextCNN(cnnconfig)


t1_int=np.argmax(y1_train,axis=1)
tag_int=np.argmax(y2_train,axis=1)


t1_dict={v:k for k,v in  text2feature.read_obj('label2num_dict_%s'%'T1').items()}
tag_dict={v:k for k,v in text2feature.read_obj('label2num_dict_%s'%'PRODUCT_TAG_NAME').items()}
 
t1_weights = compute_class_weight('balanced', np.unique(t1_int), t1_int)
t1_weights_dict = dict(zip(np.unique(t1_int), t1_weights))

tag_weights = compute_class_weight('balanced', np.unique(tag_int), tag_int)
tag_weights_dict = dict(zip(np.unique(tag_int), tag_weights))
#
#t1_name_dict={t1_dict[k]:v  for k,v in t1_weights_dict.items()}
#tag_name_dict={tag_dict[k]:v  for k,v in tag_weights_dict.items()}


model=TextCNN(cnnconfig)
model.build_model()
model.train(x_train_padded_seqs, [y1_train,y2_train],x_val_padded_seqs, [y1_val,y2_val],w_sp_train=w_sp_train.values,w_class_train=[t1_weights_dict,tag_weights_dict] ) ##data_train['sample_w'].values






















