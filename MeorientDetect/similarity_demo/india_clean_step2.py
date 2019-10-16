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

import warnings
warnings.filterwarnings('ignore')
import sys
from os import path
sys.path.append( path.dirname(path.dirname(path.abspath(__file__))))
from keras.preprocessing.text import Tokenizer
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

import re
from string import punctuation
from nltk.stem import PorterStemmer
import random
from nltk.corpus import stopwords
stop_words_dict={v:1 for v in stopwords.words('english')}

from gensim import corpora, models, similarities
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
 

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

def filter_stop_and_stem_line(stemer, tail_dict,limit_list, sentence):
    line_list=sentence.split(' ')
    line2_list=[w  for w in line_list if tail_dict.get(w) is None ]
    ret_list=line2_list  if ( (len(line2_list)>=2) or (len(line2_list)==1 and len(line2_list[0])>1) )   else line_list
    
    ret_list=[ stemer.stem(w).upper() for w in ret_list]
    return ' '.join(ret_list)



#filter_stop_and_stem_line(tail_dict,col[0])

    
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

#    line='A.TOSH 23 235 234 & SONS INDIA LTD.' 
#    line='0000SCORODITE STAINLESS INDIA PVT LTD'
    line=line.upper()
    ###drop head M/S
    line=re.sub('(^\s*)|(\s*$)','',line) ##delete head tail space
    line=re.sub('(^M/S)',' ',line)  
   
    ###drop number
    line=re.sub('[0-9]{1,100}','',line)  
    ###drop punctuation
    add_punc = '.,;《》？！’ ” “”‘’@#￥% … &×（）——+【】{};；●，。°&～、|:：\n'
    punc = punctuation + add_punc
    line=re.sub(r"[{}]+".format(punc),' ',line)  ##delete dot

    ###drop head ,tail ,mid space
    line=re.sub('[\s]*[\s]',' ',line)  ##drop multi space
    line=re.sub('(^\s*)|(\s*$)','',line) ##delete head tail space
    return line


def text_clean_batch(stemer,tail_dict,limit_list, k,line_list, return_dict):
    '''worker function'''
    ret_list=[]
    for sentence in line_list:
        sentence=noise_clean_line(sentence)
        line=filter_stop_and_stem_line(stemer, tail_dict,limit_list, sentence)
        line=line.replace(' ','')
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
            p = multiprocessing.Process(target=self.worker, args=(stemer, self.tail_dict,self.limit_list,k,line_list,return_dict))
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
            


###high frequent word######################
##################
def high_frequent(col):
    tokenizer=Tokenizer(num_words=None, filters='',lower=True,split=' ',char_level=False)  
    tokenizer.fit_on_texts(col)
    
    voc_pd=pd.DataFrame([[k,v] for (k,v) in zip(tokenizer.word_docs.keys(),tokenizer.word_docs.values())] ,columns=['key','count'])
    voc_pd=voc_pd.sort_values('count',ascending=False)
    voc_pd['pct']=voc_pd['count'].cumsum()/voc_pd['count'].sum()
    voc_pd.index=range(len(voc_pd))
    ret=voc_pd[voc_pd['count']>2000]['key'].tolist()
    ret=[v for v in ret if len(v)>1]

    return ret,voc_pd



###########################################################
data=pd.read_csv('../data/output/inda_clean_td_step1.csv')
data['gs']=np.where(data['SOURCE']=='wlq',1,0)

print('data orig len',len(data))
md=data.groupby('COMPANY_CLEAN')[['number']].sum().reset_index()
md=md[md['COMPANY_CLEAN']!=' ']
md.index=range(len(md))

print('clean step1 len',len(md))

limit_pd=pd.read_excel('../data/input/印度尾缀3.xlsx').iloc[:,:2]
limit_pd.columns=['NAME_ORIG','NAME_NEW']

tail_list=[]
for v in   limit_pd['NAME_ORIG'].apply(lambda x:x.split(' ') ) :
    tail_list.extend(v)

tail_dict={v:1  for v in set(tail_list) if len(v)>1}

limit_pd['orig_match']=limit_pd['NAME_ORIG'].apply(lambda x:'( %s$)'%x)
limit_list=limit_pd[['orig_match','NAME_NEW']].values.tolist()
#tail_dict={v:k for k,v in enumerate(set(limit_pd['NAME_NEW'])) }

col=md['COMPANY_CLEAN'].tolist()
myprocess=multiProcess(col,tail_dict,limit_list,worker=text_clean_batch)
myprocess.set_tobatch(to_batch_data)
md['text']=myprocess.run()  

data=pd.merge(data,md[['COMPANY_CLEAN','text']],on=['COMPANY_CLEAN'],how='left')
data=data.fillna('')

td_data=data[data['SOURCE']=='bill']
gs_data=data[data['SOURCE']=='wlq']

gs_uc=gs_data.groupby('text').head(1)
gs_uc.index=range(len(gs_uc))

td_uc=td_data.groupby('text').head(1)
td_uc.index=range(len(td_uc))


##################

class FeatureProc(object):
    def __init__(self):
        pass
    
    def fit_transform(self,X):
        self.tokenizer=Tokenizer(num_words=None, filters='',lower=True,split=' ',char_level=True)  
        self.tokenizer.fit_on_texts(X)
        
        voc_pd=pd.DataFrame([[k,v] for (k,v) in zip(self.tokenizer.word_docs.keys(),self.tokenizer.word_docs.values())] ,columns=['key','count'])
        self.voc_pd=voc_pd.sort_values('count',ascending=False)
        
        self.vectorizer = CountVectorizer(analyzer='char')
        #计算个词语出现的次数
        X_mat = self.vectorizer.fit_transform(X)
        #获取词袋中所有文本关键词
        self.word = self.vectorizer.get_feature_names()
        
        #类调用
        self.transformer = TfidfTransformer()
        #将词频矩阵X统计成TF-IDF值
        tfidf = self.transformer.fit_transform(X_mat)
        
        return tfidf.toarray()
    
    
    def tranform(self,X):
        X_mat = self.vectorizer.transform(X)
        #获取词袋中所有文本关键词
        tfidf = self.transformer.transform(X_mat)
        #将词频矩阵X统计成TF-IDF值
        return tfidf.toarray()
        

###############
#col=['Z S N','ZSN']
#aa = vectorizer.transform(col).toarray()

#
#xx=X.toarray()

ftproc=FeatureProc()
ft_train=ftproc.fit_transform(gs_uc['text'])

ft_pred=ftproc.tranform(td_uc['text'])



########################

import faiss
ngpus = faiss.get_num_gpus()
print("number of GPUs:", ngpus)


#mat = faiss.PCAMatrix (mt_dim,mt_dim)
#mat.train(mt)
#assert mat.is_trained
#xb = mat.apply_py(mt)

#########################


class simProc(object):
    def __init__(self,nlist=1000,k=3,):
        self.nlist = nlist               #聚类中心的个数
        self.k = k
    def fit(self,ft_train):
        train_mt=ft_train.astype('float32')
        d=train_mt.shape[1]
        ################################################33
        quantizer = faiss.IndexFlatL2(d)  # the other index
        cpu_index = faiss.IndexIVFFlat(quantizer, d, self.nlist, faiss.METRIC_INNER_PRODUCT)
        
        assert not cpu_index.is_trained
        cpu_index.train(train_mt)
        assert cpu_index.is_trained
        
        ##gpu_index = faiss.index_cpu_to_all_gpus(cpu_index) # build the index
        self.gpu_index=cpu_index
        self.gpu_index.add(train_mt)              # add vectors to the index
        print(self.gpu_index.ntotal)

    def transform(self,ft_pred): 
        pred_mt=ft_pred.astype('float32')
        print('searching................')
        t1=time.time()
        D, I = self.gpu_index.search(pred_mt, self.k) # actual search
        t2=time.time()
        print('search takes time///',t2-t1)
        return D,I


############
#
############
simproc=simProc(nlist=1000,k=3)
simproc.fit(ft_train)
D,I=simproc.transform(ft_pred)

limit_thresh=0.99
idx_bad=D<limit_thresh
D[idx_bad]=-1
I[idx_bad]=-1


print('idx to sentence //////////////////////')
gs_int_str_dict=gs_uc.loc[:,'COMPANY_CLEAN'].to_dict()
gs_int_num_dict=gs_uc.loc[:,'number'].to_dict()

def idx2np(I,D,int_num_dict,int_str_dict):
    data_list=[]
    num_list=[]
    for idx_r,line in enumerate(I):
        temp_list=[]
        tmp_list=[]
        for idx_c,v in enumerate(line):
            if v>=0:
                sentence_values=int_str_dict[v]
                num_values=int_num_dict[v]
            else:
                sentence_values=''
                num_values=0
            temp_list.append(sentence_values)
            tmp_list.append(num_values)
        data_list.append(temp_list)
        num_list.append(tmp_list)
    return data_list,num_list
    

       
data_list,num_list=idx2np(I,D,gs_int_num_dict,gs_int_str_dict)
#########################################

name_pd=pd.DataFrame(data_list)
num_pd=pd.DataFrame(num_list)

td_uc['pred_by_gs']=name_pd.iloc[:,0]
td_uc['sim_by_gs_score']=D[:,0]


#aa=td_uc[(td_uc['sim_score']<1 )&( td_uc['sim_score']>-1)]

#
#td_uc.to_csv('../data/input/td_uc_step1.csv',index=False)
#
#
#pct=td_uc[td_uc['pred_by_gs']!=''].shape[0]/len(td_uc)


#################################









####################################

simproc2=simProc(k=30)
simproc2.fit(ft_pred)
D2,I2=simproc2.transform(ft_pred)

limit_thresh=0.99
idx_bad=D2<limit_thresh
D2[idx_bad]=-1
I2[idx_bad]=-1

td_int_str_dict=td_uc.loc[:,'COMPANY_CLEAN'].to_dict()
td_int_num_dict=td_uc.loc[:,'number'].to_dict()
    
td_data_list,td_num_list=idx2np(I2,D2,td_int_num_dict,td_int_str_dict)
#########################################

td_name_pd=pd.DataFrame(td_data_list)
td_num_pd=pd.DataFrame(td_num_list)

idx_best=np.argmax(td_num_pd.values,axis=1)

td_uc['clean_best']=[v[idx_]  for (v,idx_) in  zip(td_name_pd.values,idx_best)]

idx_td=td_uc['pred_by_gs']==''
td_uc['pred_by_td']=np.where(idx_td,td_uc['clean_best'],'')


td_uc['pred']=td_uc['pred_by_gs']+td_uc['pred_by_td']

del td_uc['clean_best']





td2_data=pd.merge(td_data,td_uc[['text','pred_by_gs','pred_by_td','pred']],on=['text'],how='left')




td2_data.to_excel('../data/output/india_td_clean.xlsx',encoding='gbk',index=False)

sd=pd.read_csv('../data/input/INSTAFINANCIALS_name.csv',index_col=0)
sd=sd[['NAME','CIN']].rename(columns={'NAME':'COMPANY_NAME'})
sd['SOURCE']='wlq'
sd['COMPANY_NAME']=sd['COMPANY_NAME'].str.replace('(^\s*)|(\s*$)','')

gs2_data=pd.merge(sd,gs_data[['COMPANY_NAME','COMPANY_CLEAN']],on=['COMPANY_NAME'],how='left')

aa=gs2_data.head(100)

#gs2_data.to_excel('../data/output/india_gs_clean.xlsx',encoding='gbk',index=False)


gs2_data.to_csv('../data/output/india_gs_clean.csv',index=False)


#gs2_data.sample(1000).info()
#
#
#aa=gs2_data[gs2_data['COMPANY_CLEAN'].isnull()]

#
#td2_data.info()
#
#
#td_uc.info()


#td_uc['label_pred2']=td_uc['clean_best'].map(td_uc.set_index('COMPANY_CLEAN')['label_pred'].to_dict() )


#idx1=td_uc[td_uc['label_pred']!=td_uc['label_pred2']]

#idx_best=np.where(idx_best==td_gs_pd.shape[1]-1,None,idx_best)
#


#td_uc['gs']=np.where(td_uc['label_pred']=='',0,1)
#
#
#
#print('idx to sentence //////////////////////')
#td_int_str_dict=td_uc.loc[:,'COMPANY_CLEAN'].to_dict()
#td_int_gs_dict=gs_uc.loc[:,'gs'].to_dict()
#
#
##int_gs_dict=md.loc[:,'gs'].to_dict()
###############################################
#td_data_list,td_gs_list=idx2np(I2,D2,td_int_gs_dict,td_int_str_dict)
#td_name_pd=pd.DataFrame(td_data_list)
#td_gs_pd=pd.DataFrame(td_gs_list)
#
#td_gs_pd.iloc[:,0]=0
#td_gs_pd.iloc[:,-1]=1
#
#idx_best=np.argmax(td_gs_pd.values,axis=1)
#idx_best=np.where(idx_best==td_gs_pd.shape[1]-1,None,idx_best)
#
#
#
#
##idx_best=[]
##for k,v in enumerate(idx_temp_best):
##    idx_best.append(I2[k,v])
#
##idx_best=np.array(idx_best)
##
#
#
#ret_list=[]
#for  line,idx in  zip(td_name_pd.values,idx_best) :
#    res=line[idx] if idx is not None else np.nan
#    ret_list.append(res)
##        
##    
#td_uc['best_clean']=ret_list
#td_uc['label_pred2']=td_uc['best_clean'].map(td_uc.set_index('COMPANY_CLEAN')['label_pred'].to_dict() )

#a=td_uc.set_index('COMPANY_CLEAN')['label_pred'].to_dict()
#
#aa=td_uc.groupby('label_pred')[['COMPANY_CLEAN']].count().reset_index()
#
#
#
#td_uc.sample(1000).info()
#
#
#td_uc[td_uc['best_clean'].notnull()].shape[0]/len(td_uc)
#
#
#

#idx_max=np.argmax(num_pd.values,axis=1)

#md['COMPANY_CLEAN2']=[line[idx]  for  line,idx in  zip(name_pd.values,idx_max) ]









#####################################
aa=td_uc.groupby('label_pred').head(1)
len(aa)/len(td_uc)



#idx_max=np.argmax(num_pd.values,axis=1)

#md['COMPANY_CLEAN2']=[line[idx]  for  line,idx in  zip(name_pd.values,idx_max) ]

#md.info()

#ret=md[['COMPANY_CLEAN','number','COMPANY_CLEAN2']].groupby('COMPANY_CLEAN2').head(1)
#ret['number_sum']=md.groupby('COMPANY_CLEAN2')['number'].sum().values
#
#data_show=pd.merge(data[['COMPANY_NAME','COMPANY_CLEAN','number','SOURCE']],md[['COMPANY_CLEAN','text','COMPANY_CLEAN2']],on=['COMPANY_CLEAN'],how='left')
#
#
##data_show=pd.merge(data_show,ret[['COMPANY_CLEAN2','number_sum']],on=['COMPANY_CLEAN2'],how='left')
#pct=len(ret)/len(md)
#print('pct////',pct)
#data_show['number']=data_show['number'].fillna(0)
##data_show.to_csv('../data/output/india_tidan_similarity_clean_%s.csv'%limit_thresh,index=False)
#
##data_show.to_excel('../data/output/india_tidan_similarity_clean_%s_v4.xlsx'%limit_thresh,index=False,encoding='gbk')
#
#td=data_show[data_show['SOURCE']=='bill']
#sd=data_show[data_show['SOURCE']=='wlq']
#
#td_uc=td.groupby('COMPANY_CLEAN2').head(1)
#sd_uc=sd.groupby('COMPANY_CLEAN2').head(1)
#
#pct_td=len(td_uc)/len(td)
#pct_sd=len(sd_uc)/len(sd)
#
#print(pct_td,pct_sd)
##
#
##td.to_excel('../data/output/india_td_similarity_v4.xlsx',index=False,encoding='gbk')
##sd.to_excel('../data/output/india_wlq_similarity_v4.xlsx',index=False,encoding='gbk')
##
#
#
#
#name_set=list(set(td_uc['COMPANY_CLEAN2']).intersection(set(sd_uc['COMPANY_CLEAN2'])))
#ret_sect=len(name_set)/len(td_uc)
#print('ret_sect',ret_sect)
#

















































