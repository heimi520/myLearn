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
import collections
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

from keras.preprocessing.text import Tokenizer
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight


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


def filter_stop_and_stem_line(stemer,sentence):

    line_list=[[w.upper(),stemer.stem(w).upper()]  for w in sentence.split(' ') ]
    return line_list

    
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

#    line='A.TOSH & SONS INDIA LTD.'    
    line=line.upper()
    ###drop head M/S
    line=re.sub('(^\s*)|(\s*$)','',line) ##delete head tail space
    line=re.sub('(^M/S)',' ',line)  
   
    
    ###drop purge number
    line=' %s '%line
    line=line.replace(' ','  ')
    line=re.sub('\s[0-9]{3,1000}\s',' ',line)  
    
    
    ###drop punctuation
    add_punc = '.,;《》？！’ ” “”‘’@#￥% … &×（）——+【】{};；●，。°&～、|:：\n'
    punc = punctuation + add_punc
#    punc=punc.replace('&','').replace(' ','')
    line=re.sub(r"[{}]+".format(punc),' ',line)  ##delete dot
    
    line=line.replace(' AND ',' ')
    
    
    ###drop head ,tail ,mid space
    line=re.sub('[\s]*[\s]',' ',line)  ##drop multi space
    line=re.sub('(^\s*)|(\s*$)','',line) ##delete head tail space
    
#    stemer=PorterStemmer()
#    filter_stop_and_stem_line(stemer,line,stop_words_dict)
    return line




def text_clean_batch(stemer,k,line_list, return_dict):
    '''worker function'''
    stemer = PorterStemmer()
    ret_list=[]
    for sentence in line_list:
        line=noise_clean_line(sentence)
#        print(line)
        line=filter_stop_and_stem_line(stemer, line)
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
            



mode='wlq'  ###td, wlq

if mode=='wlq':
    dd=pd.read_excel('../data/input/阿联酋_WLQ .xlsx')
    dd.columns=['COMPANY_NAME']

elif mode=='td':
    dd=pd.read_excel('../data/input/阿联酋（全）.xlsx')
    dd.columns=['COMPANY_NAME','number']
    td_data=dd.copy()
    dd=dd[['COMPANY_NAME']]

    

dd=dd[dd['COMPANY_NAME'].notnull()]

dd=dd.drop_duplicates()


#dd=pd.DataFrame([['3 I SHIPPING & LOGISTICS PVT LTD ']],columns=['COMPANY_NAME'])

col=dd['COMPANY_NAME'].tolist()
myprocess=multiProcess(col,worker=text_clean_batch)
myprocess.set_tobatch(to_batch_data)
col=myprocess.run()  

def extract_stem_line(col):
    col_str_list=[]
    for line_list in col:
        line_str=' '.join(v[1] for v in  line_list)
        col_str_list.append(line_str)
    return col_str_list

stem_col=extract_stem_line(col)
dd['STEM_LINE']=stem_col
dd['list']=col

count_pd=dd.groupby('STEM_LINE')['STEM_LINE'].count()

#line=' asdf 234 a23 LIMIT'
#re.sub('( %s$)'%x,'',line) 

limit_pd=pd.read_excel('../data/input/尾缀0926.xlsx').iloc[:,:2]
#limit_pd=pd.read_excel('../data/input/印度尾缀3.xlsx').iloc[:,:2]
limit_pd.columns=['NAME_ORIG','NAME_NEW']
limit_pd['orig_match']=limit_pd['NAME_ORIG'].apply(lambda x:'( %s$)'%x)
limit_list=limit_pd[['orig_match','NAME_NEW']].values.tolist()

def drop_limit(limit_list,line):
    for v in limit_list:
        len_1=len(line)
        line=re.sub(v[0],' %s'%v[1],line)
        len_2=len(line)
        if len_2!=len_1:
            break
    return line

limit_dict=limit_pd.set_index('NAME_ORIG')['NAME_NEW'].to_dict()

###stem to word#####################################
doc_list=[]
for k,v in enumerate(dd.groupby('STEM_LINE')):
    if k%10000==0:
        print('k////',k)
    
    line=v[0]
    td=v[1]
    def stem2doc(td,stem_line):
        line_list=[]
        for v in td['list'].tolist():
            line_list.extend(v)
         
        stem_dict={}    
        for line in line_list:
            stem=line[1]
            word=line[0]
            if stem_dict.get(stem) is None:
                stem_dict[stem]=[]
            stem_dict[stem].append(word)
        
        stem_word_dict={}
        for k,v in stem_dict.items():
            stem_word_dict[k]=collections.Counter(v).most_common(1)[0][0]
            
       
#        new_line=' '.join([  limit_dict.get(stem_word_dict[v],stem_word_dict[v])   for v  in  stem_line.split(' ')])  
        new_line=' '.join([stem_word_dict[v]   for v  in  stem_line.split(' ')])  
        return new_line     
         
    
    
    
    doc_line=stem2doc(td,line)
    doc_line=drop_limit(limit_list,doc_line)
    doc_list.append([line, doc_line])
        

#####################

back_pd=pd.DataFrame(doc_list,columns=['STEM_LINE','COMPANY_CLEAN'])
back2_pd=back_pd[back_pd['COMPANY_CLEAN'].apply(lambda x:len(x)>5)]

#back2_pd=back_pd

res_pd=pd.merge(dd[['COMPANY_NAME','STEM_LINE']],back2_pd,on=['STEM_LINE'],how='left' )

#td_pd=pd.read_excel('../data/input/印度公司名+提单数量（251）.xlsx')
#td_pd.columns=['COMPANY_NAME','number']
#td_pd['COMPANY_NAME']=td_pd['COMPANY_NAME'].str.replace('(^\s*)|(\s*$)','')
#td_pd.to_csv('../data/input/india_clean_td.csv',index=False)


if mode=='td':
    td_data['COMPANY_NAME']=td_data['COMPANY_NAME'].str.replace('(^\s*)|(\s*$)','')
    sum_pd=td_data.groupby('COMPANY_NAME')[['number']].sum().sort_values('number',ascending=False).reset_index()
        
    res_pd['COMPANY_NAME']=res_pd['COMPANY_NAME'].str.replace('(^\s*)|(\s*$)','')
    result_pd=pd.merge(res_pd,sum_pd,on=['COMPANY_NAME'],how='left')
    result_pd['number']=result_pd['number'].fillna(0)
    
    result_pd.to_excel('../data/output/阿联酋_clean_%s.xlsx'%mode,index=False)
    
    
else:
    res_pd.to_excel('../data/output/阿联酋_clean_%s.xlsx'%mode,index=False)
    
# 







