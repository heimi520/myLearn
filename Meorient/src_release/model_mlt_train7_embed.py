# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:05:29 2019

@author: Administrator
"""

#watch -n 10 nvidia-smi


import warnings
warnings.filterwarnings('ignore')
import os
import sys
from os import path
sys.path.append( path.dirname(path.dirname(path.abspath(__file__))))

from model_config import *
from mylib.model_meorient7 import *
import gensim

import pandas as pd
from sklearn.model_selection import train_test_split


tag_args=TagModelArgs()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=tag_args.GPU_DEVICES # 使用编号为1，2号的GPU 
if tag_args.GPU_DEVICES=='-1':
    logger.info('using cpu')
else:
    logger.info('using gpu:%s'%tag_args.GPU_DEVICES)


import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import random


sub_data=pd.read_csv('../data/input/data_all_orig.csv')

logger.info('total_data shape//////%s'%sub_data.shape[0])
text2feature=Text2Feature(seq_len=tag_args.SEQ_LEN,max_words=tag_args.MAX_WORDS, \
                          is_rename_tag=False,is_add_data=False,model_id=tag_args.MODEL_ID)
sub_data['text']=text2feature.text_clean(sub_data['PRODUCT_NAME'])
sub_data['sentence']=sub_data['text'].apply(lambda x:x.split(' '))
sentences=sub_data['sentence'].tolist()

model = gensim.models.Word2Vec(sentences, min_count=5,size=tag_args.VOC_DIM,window=2,workers=8,iter=10)
data_model_dir=os.path.join('../data/temp' ,tag_args.MODEL_ID) 
model_path=os.path.join(data_model_dir,'word2vec.model')
model.save(model_path)

model.most_similar('wifi')

#model.accuracy(sentences[:100])

#
#aa=sentences[:1000]








