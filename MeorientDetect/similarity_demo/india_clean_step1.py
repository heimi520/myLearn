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



def drop_limit(limit_list,line):
    for v in limit_list:
        len_1=len(line)
        line=re.sub(v[0],' %s'%v[1],line)
        len_2=len(line)
        if len_2!=len_1:
            break
    return line.split(' ')



def filter_stop_and_stem_line(tail_dict,limit_list, sentence):
    line_list=sentence.split(' ')
    h_list=line_list[:-3]
    t_list=line_list[-3:]
    t_list=drop_limit(limit_list,' '.join(t_list))
    line_list=h_list+t_list
    return ' '.join(line_list)

    
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

#    line='A.TOSH & SONS INDIA LTD. 23 234 l.m.t'    
    line=line.upper()
    ###drop head M/S
    line=re.sub('(^\s*)|(\s*$)','',line) ##delete head tail space
    line=re.sub('(^M/S)',' ',line)  
   
    ###drop purge number
    line=' %s '%line
    line=line.replace(' ','  ')
    line=re.sub('\s[0-9]{3,1000}\s',' ',line)  
    
    ##replace & to AND
    line=re.sub('&','AND',line)  
    line=re.sub('\.','',line)  ##delete dot
    
    ###drop punctuation
    add_punc = '.,;《》？！’ ” “”‘’@#￥% … &×（）——+【】{};；●，。°&～、|:：\n'
    punc = punctuation + add_punc
    punc=punc.replace(' ','')
    line=re.sub(r"[{}]+".format(punc),' ',line)  ##delete dot
    
    
    ###drop head ,tail ,mid space
    line=re.sub('[\s]*[\s]',' ',line)  ##drop multi space
    line=re.sub('(^\s*)|(\s*$)','',line) ##delete head tail space
    
    return line


def text_clean_batch(tail_dict,limit_list,k,line_list, return_dict):
    '''worker function'''
    ret_list=[]
    for sentence in line_list:
        line=noise_clean_line(sentence)
        line=filter_stop_and_stem_line(tail_dict,limit_list, line)
        ret_list.append(line)

    return_dict[k] = ret_list



class multiProcess(Process):
    def __init__(self,data,tail_dict,limit_list,worker):
        super().__init__()
        self.col=data
        self.tail_dict=tail_dict
        self.limit_list=limit_list
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
            p = multiprocessing.Process(target=self.worker, args=(self.tail_dict,self.limit_list,k,line_list,return_dict))
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
            





mode='td'  ##td,wlq

if mode=='td':
    dd1=pd.read_csv('../data/input/印度.csv',encoding='utf-8')##.head(1000)
    dd2=pd.read_csv('../data/input/印度2.csv',encoding='utf-8')##.head(1000)
    md=pd.concat([dd1,dd2],axis=0)
    md['SOURCE']='bill'
    
#    sd=pd.read_csv('../data/input/India_WLQ(数据源).csv')
#    sd01=pd.read_excel('../data/input/INSTAFINANCIALS_name.xlsx',sheet_name=0)
#    sd02=pd.read_excel('../data/input/INSTAFINANCIALS_name.xlsx',sheet_name=1,header=None)
#    sd02.columns=sd01.columns
#    sd03=sd02[sd02['NAME'].notnull()]
#    sd=pd.concat([sd01,sd03],axis=0)
#    sd.to_csv('../data/input/INSTAFINANCIALS_name.csv')
#    
    sd=pd.read_csv('../data/input/INSTAFINANCIALS_name.csv',index_col=0)
    sd=sd[['NAME']].rename(columns={'NAME':'COMPANY_NAME'})
    sd['SOURCE']='wlq'
    
    dd=pd.concat([md,sd],axis=0)
    
    
    aa=sd.groupby('CIN')[['NAME']].count().reset_index()
    
    bb=aa[aa['NAME']>1]
    
    
#    len(aa[aa['NAME']>1])/len(aa)
#    
#    
#    .sort_values('CIN',ascending=False)
#    
    
    
#    bb=sd[sd['NAME']=='PITON SYSTEMS PRIVATE LIMITED']
#    bb2=sd[sd['NAME']=='1 POINT SYSTEMS PRIVATE LIMITED']
#    
#    sd.groupby('CID')
#   
elif mode=='wlq':
    dd=pd.read_csv('../data/input/India_WLQ(数据源).csv')
    dd_backup=dd.copy()


    

dd=dd.loc[dd['COMPANY_NAME'].notnull()]
dd=dd.drop_duplicates()


##############
limit_pd=pd.read_excel('../data/input/印度尾缀3.xlsx').iloc[:,:2]
limit_pd.columns=['NAME_ORIG','NAME_NEW']
limit_pd['orig_match']=limit_pd['NAME_ORIG'].apply(lambda x:'( %s$)'%x)
limit_list=limit_pd[['orig_match','NAME_NEW']].values.tolist()
tail_dict={v:k for k,v in enumerate(set(limit_pd['NAME_NEW'])) }

#####################################
col=dd['COMPANY_NAME'].tolist()

myprocess=multiProcess(col,tail_dict,limit_list,worker=text_clean_batch)
myprocess.set_tobatch(to_batch_data)
myprocess.set_tobatch(to_batch_data)
dd['COMPANY_CLEAN']=myprocess.run()  

md=dd.groupby('COMPANY_CLEAN').head(1)

if mode=='td':
    td_pd=pd.read_csv('../data/input/india_clean_td.csv').fillna(0)
    td_pd['COMPANY_NAME']=td_pd['COMPANY_NAME'].str.replace('(^\s*)|(\s*$)','')
    sum_pd=td_pd.groupby('COMPANY_NAME')[['number']].sum().sort_values('number',ascending=False).reset_index()
    
    dd['COMPANY_NAME']=dd['COMPANY_NAME'].str.replace('(^\s*)|(\s*$)','')
    result_pd=pd.merge(dd,sum_pd,on=['COMPANY_NAME'],how='left')
    result_pd['number']=result_pd['number'].fillna(0)
    result_pd.to_csv('../data/output/inda_clean_%s_step1.csv'%mode,index=False)
#    result_pd.to_excel('../data/output/inda_clean_%s.xlsx'%mode,index=False)
    
#    
#    aa=result_pd.sample(1000)
#    
#    ud=result_pd.groupby('COMPANY_CLEAN').head(1)
#    
#    aa=result_pd.sample(1000)
#    
elif mode=='wlq':
    res_pd['COMPANY_NAME']=res_pd['COMPANY_NAME'].str.replace('(^\s*)|(\s*$)','')
    dd_backup['COMPANY_NAME']=dd_backup['COMPANY_NAME'].str.replace('(^\s*)|(\s*$)','')

    res2_pd=res_pd.groupby(['COMPANY_NAME']).head(1)
    result_pd=pd.merge(dd_backup,res2_pd,on=['COMPANY_NAME'],how='left')
#    result_pd.to_excel('../data/output/inda_clean_%s.xlsx'%mode,index=False)
    result_pd.to_csv('../data/output/inda_clean_%s_step1.csv'%mode,index=False)




#dd2_backup=dd_backup.groupby('COMPANY_NAME').head(1)
#
#aa=dd2_backup[['COMPANY_NAME']].drop_duplicates()

#aa=result_pd.head(1000)
#bb=res2_pd.head(1000)
#
#name='	ALLEGRO VENTURES INDIA PRIVATE LIMITED'
#
#
#mm=res2_pd[res2_pd['COMPANY_NAME']==name]
#
#mm2=res_pd[res_pd['COMPANY_NAME']==name]
#
#mm3=dd[dd['COMPANY_NAME']==name]
















