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
            


tag_pd=pd.read_csv('../data/tag_class/tag_rename_v4.csv')
tag_dict=tag_pd.set_index('PRODUCT_TAG_NAME_ORIG')['PRODUCT_TAG_NAME'].to_dict()

data_his=pd.read_csv('../data/input/ali_amazon_his.csv')
data_test=pd.read_csv('../data/input/ali_amazon_test.csv')

data=pd.concat([data_his,data_test],axis=0)
data['PRODUCT_TAG_NAME']=data['PRODUCT_TAG_NAME'].map(tag_dict)
data['PRODUCT_TAG_ID']=data['PRODUCT_TAG_NAME'].copy()


textdata=data[['PRODUCT_NAME']]
textdata.columns=['text']
textdata['T1']=''
textdata['PRODUCT_TAG_NAME1']=''
textdata['PRODUCT_TAG_NAME2']=''
textdata['PRODUCT_TAG_NAME3']=''
textdata=textdata[['T1','PRODUCT_TAG_NAME1','PRODUCT_TAG_NAME2','PRODUCT_TAG_NAME3','text']]

textdata.to_csv('../data/output/text_data.csv',index=False)


 
random.seed(1)

cols_list=['BUYSELL', 'PRODUCT_NAME', 'PRODUCT_TAG_ID','PRODUCT_TAG_NAME','T1','sample_w', 'source']
data=data[cols_list]
###################
if 1==0:
    tokenizer=Tokenizer(num_words=None, filters='',lower=True,split=' ')  
    tokenizer.fit_on_texts(data['PRODUCT_NAME'])
    
    voc_pd=pd.DataFrame([[k,v] for (k,v) in zip(tokenizer.word_docs.keys(),tokenizer.word_docs.values())] ,columns=['key','count'])
    voc_pd=voc_pd.sort_values('count',ascending=True)
    voc_pd['pct']=voc_pd['count'].cumsum()/voc_pd['count'].sum()

    voc_list=voc_pd[voc_pd['pct']<=0.05]['key'].tolist()
    other_list=[]
    for v in range(5000):
        num=random.randint(1,5)
        sentence=' '.join(random.sample(voc_list,num))
#        print(sentence)
        other_list.append(sentence)
        
    other=pd.DataFrame(other_list,columns=['PRODUCT_NAME'])    
    other['BUYSELL']='sell'
    other['source']='other'
    other['T1']='T1_Other'
    other['PRODUCT_TAG_ID']='Other'
    other['PRODUCT_TAG_NAME']='Other'
    other['sample_w']=1
    
    
    def get_confuse():
        cf=pd.read_csv('../data/tag_class/word_confuse.csv')
        line_list=[]
        for v in cf.values:
            t1=v[0]
            sentence=v[2].replace('\n','')
            for vv in sentence.split(','):
                vv=re.sub('(^\s*)|(\s*$)','',vv)
                if len(re.findall('[a-zA-Z0-9]',vv))>0:
                    line_list.append([t1,v[1],vv])
        ret_list=[]
        for v in range(2000):
            ret_list.append(random.sample(line_list,1)[0])
    
        confuse_pd=pd.DataFrame(ret_list,columns=['T1','PRODUCT_TAG_NAME','PRODUCT_NAME'])
        confuse_pd['PRODUCT_TAG_ID']=confuse_pd['PRODUCT_TAG_NAME'].copy()
        confuse_pd['BUYSELL']='sell'
        confuse_pd['sample_w']=1
        confuse_pd['source']='other'
        return confuse_pd
     
    confuse_pd=get_confuse()
    other_total=pd.concat([other,confuse_pd],axis=0)
    other_total.to_csv('../data/input/my_other_v5.csv', index=False)
   
    tokenizer=Tokenizer(num_words=None, filters='',lower=True,split=' ')  
    tokenizer.fit_on_texts(other['PRODUCT_NAME'])
      
    voc_pd=pd.DataFrame([[k,v] for (k,v) in zip(tokenizer.word_docs.keys(),tokenizer.word_docs.values())] ,columns=['key','count'])
    voc_pd=voc_pd.sort_values('count',ascending=False)
    voc_pd['pct']=voc_pd['count'].cumsum()/voc_pd['count'].sum()
    voc_pd.index=range(len(voc_pd))
#        
    ######################################################################################    
    import random
    all_list=[]
    for k,v in   enumerate(data.groupby('PRODUCT_TAG_NAME')):
        name=v[0]
        td=v[1]
        print(k,name)
        ###################
        tokenizer=Tokenizer(num_words=None, filters='',lower=True,split=' ')  
        tokenizer.fit_on_texts(td['PRODUCT_NAME'])
        
        voc_pd=pd.DataFrame([[k,v] for (k,v) in zip(tokenizer.word_docs.keys(),tokenizer.word_docs.values())] ,columns=['key','count'])
        voc_pd=voc_pd.sort_values('count',ascending=False)
#        voc_pd['pct']=voc_pd['count'].cumsum()/voc_pd['count'].sum()
#        
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
        sentence_list=td['PRODUCT_NAME'].str.lower().tolist()
        keep_prob_list=[0.95,0.5,0.3,0.2]
        
        for v in range(num_last):
            sentence=random.sample(sentence_list,1)[0]
            ret_list=[]
            for word in sentence.split(' '):
                for k,(keep_prob_, cut_) in enumerate(zip(keep_prob_list, [cut1_dict,cut2_dict,cut3_dict,cut4_dict])):
                    if cut_.get(word)==1:
                        break
                if random.random()<keep_prob_:
                    ret_list.append(word)
            sentence_new=' '.join(ret_list)

            line_new[1]=sentence_new
            line_new[-2]=0.9   ####add data sample weight
            line_new[-1]='add'
            all_list.append(line_new.copy())

    data2=pd.DataFrame(all_list,columns=data.columns)
    data2.to_csv('../data/input/data_add_v5.csv',index=False)
    


other_total=pd.read_csv('../data/input/my_other_v5.csv')
data2=pd.read_csv('../data/input/data_add_v5.csv')
total_data=pd.concat([data2[cols_list],other_total[cols_list]],axis=0)
total_data=total_data[total_data['PRODUCT_NAME'].notnull()]

total_data.info()

total_data['source'].unique()

#