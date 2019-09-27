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

import gensim

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
            


ok_other=pd.read_csv('../data/input/tagpack_ok_other_step1.csv')



#other=ok_other[ok_other['T1']=='Other_T1']
#ok=ok_other[ok_other['T1']!='Other_T1']


###################
data_ok=ok_other[ok_other['T1']!='Other_T1']
cols_list=['PRODUCT_NAME','T1','T2','PRODUCT_TAG_NAME','sample_w', 'source']
data_ok=data_ok[cols_list]
data_ok['PRODUCT_NAME']=data_ok['PRODUCT_NAME'].str.lower()

aa=data_ok[data_ok['sample_w']<1]

#aa_count=data_ok.groupby('')['T1'].count().sort_values().to_frame('count')
#aa_count=data_ok.groupby('PRODUCT_TAG_NAME')['T1'].count().sort_values().to_frame('count')
#
other_data=ok_other[ok_other['T1']=='Other_T1']


other_data['T2'].unique()

import random
from nltk.corpus import stopwords
stop_words_dict={v:1 for v in stopwords.words('english')}

def sample_sentence(word_list,keep_prob_list,cut1_dict,cut2_dict,cut3_dict,cut4_dict,stop_words_dict):
    ret_list=[]
    for word in word_list:
        word=word.lower().replace(' ','')
        for k,(keep_prob_, cut_) in enumerate(zip(keep_prob_list, [cut1_dict,cut2_dict,cut3_dict,cut4_dict])):
            if cut_.get(word)==1 and stop_words_dict.get(word) is None:
                break
        if random.random()<keep_prob_:
            ret_list.append(word)
    return ret_list

        
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
    
    sub_list=voc_pd['key'].tolist()
    cut1_dict={v:1 for v in sub_list[:10]}
    cut2_dict={v:1 for v in sub_list[10:30]}
    cut3_dict={v:1 for v in sub_list[30:60]}
    cut4_dict={v:1 for v in sub_list[60:]}
 
    md=td.values.tolist()
    all_list.extend(md)
    num_last=min(max(len(md)*2,1000),3000)-len(td)

    line=md[0]     
    line_new=line.copy()
    
    keep_prob_list=[0.99,0.8,0.3,0.2]
    
    my_list=td[['PRODUCT_NAME','source']].values.tolist()
    for v in range(num_last):
        line_sample=random.sample(my_list,1)[0]
        sentence=line_sample[0]
        source=line_sample[-1]
        
        word_list=sentence.split(' ')
        if len(word_list)>=5:
            while True:
                ret_list=sample_sentence(word_list,keep_prob_list,cut1_dict,cut2_dict,cut3_dict,cut4_dict,stop_words_dict)
                if len(ret_list)>=3:
                    break          
        else:
            ret_list=word_list
            
        if len(ret_list)>0:
            sentence_new=' '.join(ret_list)
            
            line_new[0]=sentence_new
            line_new[-2]=0.8 if source=='step1_pred' else 0.85
            line_new[-1]='add' 
           
            all_list.append(line_new.copy())

data2=pd.DataFrame(all_list,columns=data_ok.columns)

print(data2.groupby('sample_w')['T1'].count())

#data2['T2'].unique()

data2.to_csv('../data/input/tagpack_add_data_step2.csv',index=False)

data2['sample_w'].unique()

other_data=other_data.sample(300000)

other_data.to_csv('../data/input/tagpack_otherdata_step2.csv',index=False)

other_data['sample_w'].unique()
 

   
    
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