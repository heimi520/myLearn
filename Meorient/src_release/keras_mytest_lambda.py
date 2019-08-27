#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 09:08:03 2019

@author: heimi
"""



import warnings
warnings.filterwarnings('ignore')


import warnings
warnings.filterwarnings('ignore')
import sys
from os import path
sys.path.append( path.dirname(path.dirname(path.abspath(__file__))))


import multiprocessing
from multiprocessing import *

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import keras.layers as layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models  import *
from keras.layers import *
import keras
from keras import *
from keras.models import load_model


from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

import re
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


from sklearn.preprocessing import LabelEncoder
from sklearn import *
from numba import jit
import os
import time
import pickle
import collections
##################################################
from keras.callbacks import ModelCheckpoint, TensorBoard,EarlyStopping, BaseLogger
import numpy as np
import pandas as pd
import os
from numpy.random import seed
import random
random.seed(1)
os.environ['PYTHONHASHSEED'] = '-1'
seed(0)  
rns=np.random.RandomState(0)
tf.set_random_seed(1)
tf.reset_default_graph()

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


data_dir='../data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
data_input_dir='../data/input'
if not os.path.exists(data_input_dir):
    os.makedirs(data_input_dir)
  
data_output_dir='../data/output'
if not os.path.exists(data_output_dir):
    os.makedirs(data_output_dir)    

data_temp_dir='../data/temp'  
if not os.path.exists(data_temp_dir):
    os.makedirs(data_temp_dir)  


from model_config import *
from mylib.model_meorient2 import *


import keras.backend.tensorflow_backend as KTF 
gpu_config=tf.ConfigProto()
gpu_config.gpu_options.per_process_gpu_memory_fraction=0.95 ##gpu memory up limit
gpu_config.gpu_options.allow_growth = True # 设置 GPU 按需增长,不易崩掉


tag_args=TagModelArgs()


tag_pd=pd.read_csv('../data/output/tag_rename_v4.csv')
tag_dict=tag_pd.set_index('PRODUCT_TAG_NAME_ORIG')['PRODUCT_TAG_NAME'].to_dict()


data_his=pd.read_csv('../data/input/ali_amazon_his.csv')
data_test=pd.read_csv('../data/input/ali_amazon_test.csv')

data=pd.concat([data_his,data_test],axis=0)
data['PRODUCT_TAG_NAME']=data['PRODUCT_TAG_NAME'].map(tag_dict)
data['PRODUCT_TAG_ID']=data['PRODUCT_TAG_NAME'].copy()

data=data[['PRODUCT_TAG_NAME','PRODUCT_TAG_ID','PRODUCT_NAME','T1','source','BUYSELL']]
data['sample_w']=1

 
data_train,data_val= train_test_split(data.head(999),test_size=0.05)
#####################################################
text2feature=Text2Feature(seq_len=tag_args.SEQ_LEN,max_words=tag_args.MAX_WORDS, is_rename_tag=False,is_add_data=False,model_id=tag_args.MODEL_ID)
x_train_padded_seqs,x_val_padded_seqs,y1_train,y1_val,y2_train,y2_val,w_sp_train=text2feature.pipeline_fit_transform(data_train,data_val)

#





place_x= Input(shape=(tag_args.SEQ_LEN,),name='x_input')

embed = keras.layers.Embedding(tag_args.MAX_WORDS , ###self.vocab_len + 1,
                               tag_args.VOC_DIM ,
                               trainable=True ,
                               weights=None , 
                               input_length=tag_args.SEQ_LEN)(place_x)


x1 = Lambda(lambda x:x[:,::-1,:],output_shape=(tag_args.SEQ_LEN,tag_args.VOC_DIM))(embed)

merge=Reshape([tag_args.SEQ_LEN*tag_args.VOC_DIM])(x1)

net=Dense(64)(merge)

import keras.backend as K

def Tag2T1(x,cut_list):    
    line_list=[]
    for k, v in enumerate(cut_list):
        if k==0:
            line=x[:,:v]
        else:
            line=x[:,v:cut_list[k-1]]
        line_list.append(line)
    line_list.append(x[:,cut_list[-1]:])
    
    
    ret_list=[]
    for line in line_list:
        max_=Reshape([1])(K.max(line,axis=1))
        ret_list.append(max_)
    
    out=keras.layers.concatenate(ret_list, axis=1)
    return out

batch_size=len(x_train_padded_seqs)

#a = Input(shape=(4,2))
#net2 = Lambda(process ) (net)
cut_list=[10,50]
net2 = Lambda(Tag2T1,arguments={'cut_list':cut_list} ) (net)
     
#net2=net   
#output=Dense(3)(net2)



model = Model(place_x, net2)

model.summary()
#model.predict({'x_input':x_train_padded_seqs})


intermediate_layer_model = Model(inputs=model.input,                                 
                                 outputs= net2)
#                                 model.get_layer('lambda_9').output
#                                 )

pred_out = intermediate_layer_model.predict({'x_input':x_train_padded_seqs})#这个数据就是原始模型的输入数据，

print('pred_out shape///',pred_out.shape)





#
#model = Model(a, output)
#x_test = np.array([[[1,2],[2,3],[3,4],[4,5]]])
#print model.predict(x_test)
#








#
#
#####session init######
#sess = tf.Session(config=gpu_config)
#sess.run(tf.global_variables_initializer())
#sess.run(tf.global_variables_initializer())
#sess.run(tf.local_variables_initializer())   
##self.sess=sess
#
#
#[d0,d1,d_out]=sess.run([embed,x1,output],feed_dict={place_x:x_train_padded_seqs})
#
#
#a0=d0[0]
#a1=d1[0]

#
#







