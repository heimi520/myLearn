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



#

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
    line=re.sub('[0-9]{1,40}[.][0-9]*',' ^ ',line) ###match decimal
    line=re.sub('mp3','mp****',line)
    line=re.sub('mp4','mp****',line)
    line=re.sub('t[\s-]*shirt',' t**shirt',line)
    line=re.sub('single[\s-]*len',' single**len',line)
 
    line=re.sub(',',';',line)
    line=re.sub('\.',';',line)
    line=re.sub(';',' ; ',line)
    
    add_punc = '.,;《》？！’ ” “”‘’@#￥% … &×（）——+【】{};；●，。°&～、|:：\n'
    punc = punctuation + add_punc
    punc=punc.replace('*','').replace(';','')
    line=re.sub(r"[{}]+".format(punc)," ",line)  ##delete dot
    line=re.sub('([2][0][0-9][0-9])','yyyy',line) ###match year 20**
    line=re.sub('\d{1,20}','*',line)  ##match int number
    
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
            

#
tag1=pd.read_excel('../data/ali_data/服装&3C 标签7.02_new.xlsx',sheet_name=0)
tag2=pd.read_excel('../data/ali_data/服装&3C 标签7.02_new.xlsx',sheet_name=1)
tag=pd.concat([tag1,tag2],axis=0)
tag['PRODUCT_TAG_NAME']=tag['Product Tag'].str.replace('(^\s*)|(\s*$)','')




data0=pd.read_csv('../data/input/data_filter.csv')
tag_pd=pd.read_csv('../data/tag_class/tag_check_pd.csv')

data=pd.merge(data0,tag_pd,on=['PRODUCT_TAG_NAME'],how='inner')
data=data[data['PRODUCT_NAME'].notnull()]
data[data.isnull()]=''

    
aa_count=data.groupby('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME'].count().sort_values(ascending=True).to_frame('count')
aa_count['tag']=aa_count.index


#aa=pd.Series(list(set(tag['PRODUCT_TAG_NAME'])-set(data['PRODUCT_TAG_NAME'])))
#
#'Bank & Airline Uniforms'  in tag_pd['PRODUCT_TAG_NAME'].tolist()


#aa=pd.Series(data0['PRODUCT_TAG_NAME'].unique().tolist())


tag_test='Portable Power Banks'
#tag_test='Batteries'
aa=data[data['PRODUCT_TAG_NAME']==tag_test]

#aa2=aa[aa['sect_flag']==1]


k=0
stemer = PorterStemmer()
return_dict={}
text_dict={}
line_list=aa[['PRODUCT_NAME','tag1','tag2','tag3','except']].values
ret_list,text_list=text_clean_batch(k,stemer,line_list, return_dict,text_dict)

aa['sect_idx']=ret_list
aa['text']=text_list
aa_ok=aa[aa['sect_idx']==1]

#aa_ok.to_csv('test.csv')
#


if 1==1:
    mp=multiProcess(data=data[['PRODUCT_NAME','tag1','tag2','tag3','except']].values,worker=text_clean_batch)
    ret_list,text_list=mp.run()
    data['sect_flag']=ret_list
    data['text']=text_list
   
    data_ok=data[data['sect_flag']==1]
    data_ok['sample_w']=1
    
    data_bad=data[data['sect_flag']==0]
    data_bad['sample_w']=0.1
    
    tag_rename=data_ok.groupby('PRODUCT_TAG_ID')['PRODUCT_TAG_ID'].count().sort_values().to_frame('count')
    tag_rename['PRODUCT_TAG_NAME_ORIG']=tag_rename.index
    tag_rename['PRODUCT_TAG_NAME']=tag_rename['PRODUCT_TAG_NAME_ORIG'].copy()
    
    tag_rename.to_csv('../data/tag_class/tag_rename_v4.csv',index=False)
#    
#    set(tag['PRODUCT_TAG_NAME'])-set(tag_rename['PRODUCT_TAG_NAME'])
#    
    #m=data_all[data_all['PRODUCT_TAG_NAME']=='Other']
    #
    aa_count=data_ok.groupby('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME'].count().sort_values(ascending=True).to_frame('count')
    aa_count['tag']=aa_count.index
    #
    
    aa_count[aa_count['count']<500].to_csv('../data/output/data_not_enough.csv')
    
    aa_count_all=data.groupby('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME'].count().sort_values(ascending=True).to_frame('count')
    aa_count_all['tag']=aa_count_all.index
    
#    aa_cc=aa_cc.sort_values('count_y',ascending=False)
#    
#    aa_cc['count_y'].plot.bar()
#    
    aa_cc=pd.merge(aa_count_all,aa_count,on=['PRODUCT_TAG_NAME'],how='left')
    aa_cc['pct']=aa_cc['count_y']/aa_cc['count_x']
    aa_cc=aa_cc.sort_values('pct')
    
#    aa=data_all[data_all['PRODUCT_TAG_NAME']=='Mobile Phones']
    
    data_his,data_test = train_test_split(data_ok, test_size=0.01)
    
    d_dict={}
    for v in data_ok.groupby('PRODUCT_TAG_NAME'):
        d_dict[v[0]]=v[1]
    
    
    
##    
    data_his.to_csv('../data/input/ali_amazon_his.csv',index=False)
    data_test.to_csv('../data/input/ali_amazon_test.csv',index=False)
    






















