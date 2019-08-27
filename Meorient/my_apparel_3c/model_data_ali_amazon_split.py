#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 13:44:20 2019

@author: heimi
"""

#watch -n 10 nvidia-smi


import warnings
warnings.filterwarnings('ignore')
import sys
from os import path
sys.path.append( path.dirname(path.dirname(path.abspath(__file__))))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from numpy.random import seed
import os
os.environ['PYTHONHASHSEED'] = '-1'
seed(0)  
rns=np.random.RandomState(0)


import re
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import multiprocessing
from multiprocessing import *

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def noise_clean_line(line):
    """
    noise clean line
    """
#    line='DSLR  SLR   DSLR  SLR Single Lens  Reflex   Film       Instant      Security      Sports    Action  Digital ## charger adapter cables   hoods bags  straps ### phone iphone smartphone watch smartwatch pc computer laptop mac ipod ipad '
    line=line.lower()
    line=re.sub("'s",'',line) ##delete 's
    line=re.sub('t[\s-]*shirt',' tshirt',line)
    line=re.sub('single[\s-]*len',' single len',line)
    
    add_punc = '.,;《》？！’ ” “”‘’@#￥% … &×（）——+【】{};；●，。°&～、|:：\n'
    punc = punctuation + add_punc

    line=re.sub(r"[{}]+".format(punc)," ",line)  ##delete dot

    line=re.sub('[\s]*[\s]',' ',line)  ##drop multi space
    line=re.sub('(^\s*)|(\s*$)','',line) ##delete head tail space
    return line

def filter_line(line,stemer):
    line=noise_clean_line(line)
    line_list=[ stemer.stem(v) for v in  line.split(' ')]
    return line_list



def text_clean_batch(k,stemer,line_list, return_dict,text_dict):
    """worker function'''
    """
        
    ret_list=[]
    text_list=[]
    for (line_0,line_1,line_2,line_3,line_9) in line_list:
#        line_0="""Olympus M.Zuiko Digital ED 75 to 300mm II F4.8-6.7 Zoom Lens, for Micro Four Thirds Cameras"""
        line0=filter_line(line_0,stemer)
        line1=filter_line(line_1,stemer)
        line2=filter_line(line_2,stemer)
        line3=filter_line(line_3,stemer)
        line9=filter_line(line_9,stemer)
        
        sect1=set(line0).intersection(set(line1))
        sect2=set(line0).intersection(set(line2))
        sect3=set(line0).intersection(set(line3))
        sect9=set(line0).intersection(set(line9))
        con1= True if len(line1)==1 and line1[0]=='' else   len(sect1)>=1
        con2= True  if len(line2)==1 and line2[0]=='' else   len(sect2)>=1
        con3= True  if len(line3)==1 and line3[0]=='' else   len(sect3)>=1
        con9= True  if len(line9)==1 and line9[0]=='' else   len(sect9)==0
        idx=int(con1  & con2 & con3 &con9) 

        ret_list.append(idx)
        text_list.append(line0)

    return_dict[k] = ret_list
    text_dict[k]=text_list
    return ret_list,text_list

    
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
        text_dict = manager.dict()
        jobs = []
        
        stemer = PorterStemmer()
        for (k,line_list) in batch_dict.items():
            logger.info('start worker//%s'%k)
            p = multiprocessing.Process(target=self.worker, args=(k,stemer,line_list,return_dict,text_dict))
            jobs.append(p)
            p.start()
            
        for k,proc in enumerate(jobs):
            logger.info('join result// %s'%k)
            proc.join()
        
        ret_list=[]
        text_list=[]
        for k in range(num):
            logger.info('k///%s /// len// %s'%(k,len(return_dict[k])))
            ret_list.extend(return_dict[k])
            text_list.extend(text_dict[k])
            
        return ret_list,text_list
            


data0=pd.read_csv('../data/input/apparel_3c_orig.csv')
data0['tag_orig']=data0['PRODUCT_TAG_NAME'].copy()

datafilter=data0[data0['T1']!='T1_Other']

reg=pd.read_csv('../data/tag_class/tag_check_pd.csv')
reg.columns=['PRODUCT_TAG_NAME','tag1','tag2','tag3','except']
reg=reg.fillna('')

data=pd.merge(datafilter,reg,on=['PRODUCT_TAG_NAME'],how='inner')
data['PRODUCT_TAG_ID']=data['PRODUCT_TAG_NAME'].copy()


tag_test='Data Cables'
#tag_test='Mattress Cover'
aa=data[data['PRODUCT_TAG_NAME']==tag_test]


k=0
stemer = PorterStemmer()
return_dict={}
text_dict={}
line_list=aa[['PRODUCT_NAME','tag1','tag2','tag3','except']].values
ret_list,text_list=text_clean_batch(k,stemer,line_list, return_dict,text_dict)

aa['sect_idx']=ret_list
aa['text']=text_list
aa_ok=aa[aa['sect_idx']==1]
#aa_ok[['PRODUCT_TAG_NAME','PRODUCT_NAME']].to_csv('../data/output/mytest.csv',index=False)

#aa_ok.to_csv('test.csv')

if 1==1:
    
    mp=multiProcess(data=data[['PRODUCT_NAME','tag1','tag2','tag3','except']].values,worker=text_clean_batch)
    ret_list,text_list=mp.run()
    data['sect_flag']=ret_list
    data['text']=text_list
       

    data_ok=data[data['sect_flag']==1]
    data_ok['sample_w']=1
    
    data_ok.info()
    
    ok_other=pd.merge(data0,data_ok[['PRODUCT_NAME','sample_w','PRODUCT_TAG_NAME']],on=['PRODUCT_NAME','PRODUCT_TAG_NAME'],how='left')
    
    ok_other.sample(10000).info()

    idx_other=ok_other['sample_w'].isnull()
    ok_other.loc[idx_other,'T2']='Other_T2'
    ok_other.loc[idx_other,'T1']='Other_T1'
    ok_other.loc[idx_other,'PRODUCT_TAG_NAME']='Other_TAG'
    ok_other.loc[idx_other,'sample_w']=0.1
    
    ok_other.sample(10000).info()

    ok_other.to_csv('../data/input/a3c_ok_other.csv',index=False)

    aa_count=data_ok.groupby('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME'].count().sort_values(ascending=True).to_frame('count')
    aa_count['tag']=aa_count.index
    aa_count['T1']=aa_count['tag'].map(data.set_index('PRODUCT_TAG_NAME')['T1'].to_dict())
    
    aa_count_all=data.groupby('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME'].count().sort_values(ascending=True).to_frame('count')
    aa_count_all['tag']=aa_count_all.index
    

    aa_cc=pd.merge(aa_count_all,aa_count,on=['PRODUCT_TAG_NAME'],how='left')
    aa_cc['pct']=aa_cc['count_y']/aa_cc['count_x']
    aa_cc=aa_cc.sort_values('pct')



##    aa=data_all[data_all['PRODUCT_TAG_NAME']=='Mobile Phones']
#    
#    data_his,data_test = train_test_split(data_ok, test_size=0.01)
#    
#    d_dict={}
#    for v in data_ok.groupby('T2'):
#        d_dict[v[0]]=v[1]
#    
#    
#    data_his.to_csv('../data/input/tagpack_his.csv',index=False)
#    data_test.to_csv('../data/input/tagpack_test.csv',index=False)
#    
#
#
#
#
#
#
#
#
#













