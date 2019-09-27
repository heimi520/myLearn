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
from keras.preprocessing.text import Tokenizer
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

import re
from string import punctuation
from nltk.stem import PorterStemmer
import random
from nltk.corpus import stopwords
stop_words_dict={v:1 for v in stopwords.words('english')}

import pandas as pd
from sklearn.model_selection import train_test_split
import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import keras 
#keras.backend.set_floatx('float16')

logger.info('keras floatx:%s'%(keras.backend.floatx()))

import multiprocessing
from multiprocessing import *

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import random


def add_data_batch(k,stemer,line_list, return_dict):
    '''worker function'''
    
    ret_list=[]
    for line in line_list:
        line_list=line.split(' ')
        random.shuffle(line_list)
        line=' '.join(line_list)
        ret_list.append(line)

    return_dict[k] = ret_list


class multiProcess(Process):
    def __init__(self,data,worker):
        super().__init__()
        self.col=data
        self.worker=worker
#        
    def set_worder(self,worker):
        self.worker=worker    
        
    def to_batch_dict(self,col,num):
        batch_size=int(len(col)/num)
        batch_dict={}
        k=0
        for k in range(num-1):
            line_list=col[k*batch_size:(k+1)*batch_size]
            batch_dict[k]=line_list
        line_list=col[(k+1)*batch_size:]
        batch_dict[k+1]=line_list
        return batch_dict
            
    def run(self):
        col=self.col
        num=cpu_count()
        batch_dict=self.to_batch_dict(col,num)
        
        manager = Manager()
        return_dict = manager.dict()
        jobs = []
        
        stemer = PorterStemmer()
        for (k,line_list) in batch_dict.items():
            logger.info('start worker//%s'%k)
            p = multiprocessing.Process(target=self.worker, args=(k,stemer,line_list,return_dict,))
            jobs.append(p)
            p.start()
            
        for k,proc in enumerate(jobs):
            logger.info('join result// %s'%k)
            proc.join()
        
        ret_list=[]
        for k in range(num):
            logger.info('k///%s /// len// %s'%(k,len(return_dict[k])))
            ret_list.extend(return_dict[k])
            
        return ret_list
            


def text_clean(col):
    add_punc = '.,;《》？！’ ” “”‘’@#￥% … &×（）——+【】{};；●，。°&～、|:：\n'
    punc = punctuation + add_punc
    line_list=[]
    for v in col:
        line=re.sub(r"[{}]+".format(punc)," ", v.lower())
        line=re.sub('[\s]*[\s]',' ',line)  ##drop multi space
        line=re.sub('(^\s*)|(\s*$)','',line) ##delete head tail space
        line_list.append(line)
    return line_list




def sample_sentence(word_list,keep_prob_list,cut1_dict,cut2_dict,cut3_dict,cut4_dict):
    ret_list=[]
    for word in word_list:
        word=word.lower().replace(' ','')
        sample_word=None
        for k,(keep_prob_, cut_) in enumerate(zip(keep_prob_list, [cut1_dict,cut2_dict,cut3_dict,cut4_dict])):
            if cut_.get(word)==1 : ###drop out stop words
                sample_word=word
                break
        ##big then keep_prob ,then drop out this word
        if random.random()>keep_prob_:
            sample_word=None
        ret_list.append(sample_word)
#        print('word',sample_word,'k',k,'keepprob',keep_prob_)
        
    ret_list=[v for v in ret_list if v is not None ]   ###drop None
#    print(ret_list)
    
    return ret_list


   

ok_other=pd.read_csv('../data/input/tagpack_ok_other2.csv')

###################
data_ok=ok_other[ok_other['T1']!='Other_T1']
cols_list=['PRODUCT_NAME','T1','T2','PRODUCT_TAG_NAME','sample_w', 'source']
data_ok=data_ok[cols_list]
other_data=ok_other[ok_other['T1']=='Other_T1']
 
all_list=[]        
for k,v in   enumerate(data_ok.groupby('PRODUCT_TAG_NAME')):
    name=v[0]
    td=v[1]
    print(k,name)
    ###################
    tokenizer=Tokenizer(num_words=None, filters='',lower=True,split=' ')  
    tokenizer.fit_on_texts(td['PRODUCT_NAME'])
    
    voc_pd=pd.DataFrame([[k,v] for (k,v) in zip(tokenizer.word_docs.keys(),tokenizer.word_docs.values())] ,columns=['key','count'])
    voc_pd=voc_pd.sort_values('count',ascending=False)
    
    sub_list=[ v  for v in  voc_pd['key'] if stop_words_dict.get(v) is  None ]
    cut1_dict={v:1 for v in sub_list[:10]}
    cut2_dict={v:1 for v in sub_list[10:30]}
    cut3_dict={v:1 for v in sub_list[30:60]}
    cut4_dict={v:1 for v in sub_list[60:]}
 
    md=td.values.tolist()
    all_list.extend(md)
    num_last=min(max(len(md)*2,1000),4000)-len(td)

    line=md[0]     
    line_new=line.copy()
    
    keep_prob_list=[0.99,0.8,0.3,0.2]
    
    tmp=td[['PRODUCT_NAME','source']].copy()
    tmp['PRODUCT_NAME']=text_clean(tmp['PRODUCT_NAME'].tolist())
    my_list=tmp.values.tolist()
    
    for v in range(num_last):
        line_sample=random.sample(my_list,1)[0]
        sentence=line_sample[0]
        source=line_sample[-1]
        
        word_list=sentence.split(' ')
        
        if len(word_list)>=5:
            
            c=0
            while True:
                c+=1
                ret_list=sample_sentence(word_list,keep_prob_list,cut1_dict,cut2_dict,cut3_dict,cut4_dict)
#                print(len(ret_list))
                if len(ret_list)>=3:
                    break    
                if c>5 and len(ret_list)>0:
                    break
        else:
            ret_list=word_list

        if len(ret_list)>0:
            sentence_new=' '.join(ret_list)
            line_new[0]=sentence_new
            line_new[-2]=0.95
            line_new[-1]='add' 
           
            all_list.append(line_new.copy())
            
    
data2=pd.DataFrame(all_list,columns=data_ok.columns)


a=data2.groupby('PRODUCT_TAG_NAME')['T1'].count().sort_values()


data2.to_csv('../data/input/tagpack2_add_data_step1.csv',index=False)

data2['sample_w'].unique()

other_data=other_data.sample(300000)
#other_data=other_data
other_data.to_csv('../data/input/tagpack2_otherdata_step1.csv',index=False)

other_data['sample_w'].unique()
 

#aa=data2[data2['PRODUCT_TAG_NAME']=='Home Washing Machines']
#aa=aa[aa['source']='add']  

 
    
#other_total=pd.read_csv('../data/input/my_other_v5.csv')
#data2=pd.read_csv('../data/input/data_add_v5.csv')
#total_data=pd.concat([data2[cols_list],other_total[cols_list]],axis=0)
#total_data=total_data[total_data['PRODUCT_NAME'].notnull()]
#
#total_data.info()
#
#total_data['source'].unique()
#
##