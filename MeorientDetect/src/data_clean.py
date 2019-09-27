#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 11:10:43 2019

@author: heimi
"""


import re
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

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
import keras.backend as K

from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

import re
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import gensim.models.word2vec as word2vec

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
logger.setLevel(logging.INFO)


limit=pd.read_csv('../data/input/limit_corp_word.csv')


stop_words_dict={v:1 for v in limit['key']}

#

def filter_stop_and_stem_line(stemer,sentence,stop_words_dict):
#    line=' '.join([stemer.stem(w)  for w in sentence.split(' ') if stop_words_dict.get(w) is  None  ])
    line=' '.join([stemer.stem(w)  for w in sentence.split(' ') ])
    return line
#filter_stop_and_stem_line('limit comp',stop_words_dict)

    
def to_batch_data(col,num):
    """
    col:list
    """
    batch_size=int(len(col)/num)
    batch_dict={}
    k=0
    if num>1:
        for k in range(num-1):
            line_list=col[k*batch_size:(k+1)*batch_size]
            batch_dict[k]=line_list
        line_list=col[(k+1)*batch_size:]
        batch_dict[k+1]=line_list
    else:
        line_list=col[k*batch_size:(k+1)*batch_size]
        batch_dict[k]=line_list
    return batch_dict





def noise_clean_line(line):
    """
    noise clean line
    """
    line=line.lower()
    
    add_punc = '.,;《》？！’ ” “”‘’@#￥% … &×（）——+【】{};；●，。°&～、|:：\n'
    punc = punctuation + add_punc
    line=re.sub(r"[{}]+".format(punc)," ",line)  ##delete dot

    line=re.sub('[\s]*[\s]',' ',line)  ##drop multi space
#    line=re.sub('[0-9]{4,100}','',line)  ##drop long num
#    line=re.sub('[0-9]{4,100}','',line)  ##drop long num
    line=re.sub('(^\s*)|(\s*$)','',line) ##delete head tail space
    return line


def text_clean_batch(stemer,k,line_list, return_dict):
    '''worker function'''
    stemer = PorterStemmer()
    ret_list=[]
    for sentence in line_list:
        line=noise_clean_line(sentence)
#        print(line)
        line=filter_stop_and_stem_line(stemer, line,stop_words_dict)
#        print(line)
        ret_list.append(line)

    return_dict[k] = ret_list




class multiProcess(Process):
    def __init__(self,data,worker):
        super().__init__()
        self.col=data
        self.worker=worker
#        
    def set_worder(self,func):
        self.worker=func
        
    def set_tobatch(self,func):
        self.to_batch_data=func

        
    def run(self):
        col=self.col
        num=cpu_count()
        batch_dict=self.to_batch_data(col,num)
        
        stemer = PorterStemmer()
        
        manager = Manager()
        return_dict = manager.dict()
        jobs = []
    
        for (k,line_list) in batch_dict.items():
            logger.info('start worker//%s'%k)
            p = multiprocessing.Process(target=self.worker, args=(stemer,k,line_list,return_dict))
            jobs.append(p)
            p.start()
            
        for k,proc in enumerate(jobs):
            logger.info('join result// %s'%k)
            proc.join()
        
        ret_list=[]
        c=0
        for k in range(num):
            c_k=len(return_dict[k])
            c+=c_k
            logger.info('k///%s /// len// %s'%(k,c_k))
            ret_list.extend(return_dict[k])
        logger.info('collect total len//%s'%(c))    
        return ret_list
            



dd1=pd.read_csv('../data/input/印度.csv',encoding='utf-8')##.head(1000)
dd2=pd.read_csv('../data/input/印度2.csv',encoding='utf-8')##.head(1000)
dd=pd.concat([dd1,dd2],axis=0)

dd=dd[dd['COMPANY_NAME'].notnull()]

dd=dd.drop_duplicates()


#dd=dd.head(1000)


col=dd['COMPANY_NAME'].tolist()
myprocess=multiProcess(col,worker=text_clean_batch)
myprocess.set_tobatch(to_batch_data)
col=myprocess.run()  

#
dd['COMPANY_CLEAN']=col


dd.to_excel('../data/output/data_orig.xlsx',index=False)



md=dd.groupby('COMPANY_CLEAN').head(1)


len(md)/(len(dd1)+len(dd2))


md.to_excel('../data/output/clean_result.xlsx',index=False)


#ret=dd[dd['COMPANY_CLEAN'].apply(lambda x:len(x)>=5)]




#ret.to_csv('../data/input/clean_data.csv',index=False)
#

























