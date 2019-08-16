#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 09:52:51 2019

@author: heimi
"""

import fasttext as ft
from sklearn.model_selection import train_test_split
import pandas as pd
import random
import os
from fasttext_config import *
import pickle

tag_pd=pd.read_csv('../data/tag_class/tag_rename_v4.csv')
tag_dict=tag_pd.set_index('PRODUCT_TAG_NAME_ORIG')['PRODUCT_TAG_NAME'].to_dict()

data_his=pd.read_csv('../data/input/ali_amazon_his.csv')
data_test=pd.read_csv('../data/input/ali_amazon_test.csv')

data=pd.concat([data_his,data_test],axis=0)
data['PRODUCT_TAG_NAME']=data['PRODUCT_TAG_NAME'].map(tag_dict)
data['PRODUCT_TAG_ID']=data['PRODUCT_TAG_NAME'].copy()


 
random.seed(1)

cols_list=['BUYSELL', 'PRODUCT_NAME', 'PRODUCT_TAG_ID','PRODUCT_TAG_NAME','T1','sample_w', 'source']
data=data[cols_list]




tag_pd=pd.read_csv('../data/tag_class/tag_rename_v4.csv')
tag_dict=tag_pd.set_index('PRODUCT_TAG_NAME_ORIG')['PRODUCT_TAG_NAME'].to_dict()

data_his=pd.read_csv('../data/input/ali_amazon_his.csv')
data_test=pd.read_csv('../data/input/ali_amazon_test.csv')

data=pd.concat([data_his,data_test],axis=0)
data['PRODUCT_TAG_NAME']=data['PRODUCT_TAG_NAME'].map(tag_dict)
data['PRODUCT_TAG_ID']=data['PRODUCT_TAG_NAME'].copy()


 
random.seed(1)

cols_list=['BUYSELL', 'PRODUCT_NAME', 'PRODUCT_TAG_ID','PRODUCT_TAG_NAME','T1','sample_w', 'source']
data=data[cols_list]


other_total=pd.read_csv('../data/input/my_other_v5.csv')
data2=pd.read_csv('../data/input/data_add_v5.csv')
total_data=pd.concat([data2[cols_list],other_total[cols_list]],axis=0)
total_data=total_data[total_data['PRODUCT_NAME'].notnull()]

total_data.info()

total_data['source'].unique()

#

#total_data.info()
#
#total_data['source'].unique()
##
#tag_id_dict={v:'A%s'%k for k,v in enumerate(set(total_data['PRODUCT_TAG_NAME']))  }
#id_tag_dict={v:k  for (k,v) in tag_id_dict.items()}
#total_data['PRODUCT_TAG_ID']=total_data['PRODUCT_TAG_NAME'].map(tag_id_dict)

from string import punctuation
import re
#add_punc = '.,;《》？！’ ” “”‘’@#￥% … &×（）——+【】{};；●，。°&～、|:：\n'
#punc = punctuation + add_punc
#
#total_data['PRODUCT_NAME']=total_data['PRODUCT_NAME'].apply(lambda line:re.sub(r"[{}]+".format(punc)," ",line.lower()))  
#total_data['key']=total_data['PRODUCT_TAG_NAME'].apply(lambda x:x.lower().replace(' ',''))
#total_data['text']=total_data['PRODUCT_NAME']+'      '+'__label__'+total_data['key']

#aa=total_data[total_data['PRODUCT_NAME'].apply(lambda x:'"' in x)]

#aa=list(total_data['key'].unique())

    
data_train,data_val= train_test_split(total_data,test_size=0.05)

input_file='../data/input/fast_train_data.csv'
test_file='../data/input/fast_test_data.csv'
#data_train[['text']].to_csv(input_file,index=False,header=None)
#data_val[['text']].to_csv(test_file,index=False,header=None)




output='../data/output'
# set params
dim=100
lr=0.01
epoch=5
min_count=1
word_ngrams=3
bucket=200000
#thread=4
silent=1
label_prefix='__label__'

import fasttext as ft


class FastText(object):
    model_name='fasttext'
    def __init__(self,config):
        self.epoch=config.EPOCH_MAX
        self.word_ngrams=config.WORD_NGRAMS
        self.data_model_dir=os.path.join('../data/temp' ,config.MODEL_ID) 
        if not os.path.isdir(self.data_model_dir):
            os.makedirs(self.data_model_dir)  
    
    def preprocess(self,td):
        add_punc = '.,;《》？！’ ” “”‘’@#￥% … &×（）——+【】{};；●，。°&～、|:：\n'
        punc = punctuation + add_punc
        td['x']=td['x'].apply(lambda line:re.sub(r"[{}]+".format(punc)," ",line.lower()))  
        return td

    def train(self,x_train,y_train):
        
        td=pd.DataFrame(x_train.values,columns=['x'])
        td['y']=y_train.values

        td['key']=td['y'].apply(lambda x:x.lower().replace(' ',''))
        key_tag_dict=td.set_index('key')['y'].to_dict()
        self.save_obj(key_tag_dict,'key_tag_dict')
        
        td=self.preprocess(td)

        td['text']='__label__'+td['key']+'  '+td['x']
        input_file='../data/input/fasttext_train.csv'
        td[['text']].to_csv(input_file,index=False,header=None)
#        
        self.model = ft.train_supervised(input_file, 
        #                                 label_prefix='__label__',
                                         epoch=50,
                                         dim=50,
                                         lr=1.0,
                                         word_ngrams=3,
                                         minn=3,
                                         maxn=5,
                                         minCount=5,
                                         minCountLabel=100, 
                                         loss='hs', ##'ova',
                                         bucket =20000000,
#                                         pretrainedVectors="../data/fasttext/wiki.en.vec"
        #                                 bucket= 2000000
#                                         verbose=1,
                                         )
        
        self.model.save_model(os.path.join(self.data_model_dir,'%s.h5'%self.model_name)) # 保存模型

    def predict_classes(self,test_list):
        key_tag_dict=self.read_obj('key_tag_dict')
#        self.model=ft.load_model(os.path.join(self.data_model_dir,'%s.h5'%self.model_name))
        result=self.model.predict(test_list)
        return [key_tag_dict[v[0].replace('__label__','')]  for v in  result[0]]
    
    
    def save_obj(self,obj,obj_name):
        with open(os.path.join(self.data_model_dir,'%s.pickle'%obj_name), 'wb') as f:
            pickle.dump(obj, f)
            
    def read_obj(self,obj_name):
        with open(os.path.join(self.data_model_dir,'%s.pickle'%obj_name), 'rb') as f:
            return  pickle.load(f)            
 

       
fasttextconfig=fastextConfig()

x_train=total_data[['PRODUCT_NAME']]
y_train=total_data['PRODUCT_TAG_NAME']
model=FastText(fasttextconfig)

#len(set(y_train))

model.train(x_train,y_train)

key_tag_dict=model.read_obj('key_tag_dict')
#tag_key_dict={v:k for  k,v in key_tag_dict.items()} 

total_data['pred_tag']=model.predict_classes(total_data['PRODUCT_NAME'].tolist())


#model.predict_classes(['Acer G257HU smidpx 25-Inch WQHD (2560 x 1440) Widescreen Monitor'])


total_data['ok']=total_data['PRODUCT_TAG_NAME']==total_data['pred_tag']


acc=total_data['ok'].sum()/len(total_data)
print('acc',acc)
  

bad=total_data[total_data['PRODUCT_TAG_NAME']!=total_data['pred_tag']]



  
#result=model.result[0]
#a=[ key_tag_dict[ v[0].replace('__label__','') ] for v in  result]
#
##for v in   result:
#    a=tag_key_dict[ v[0].replace('__label__','') ]
#    


#test = classifier.test(test_file) # 输出测试结果
#classifier.get_labels() # 输出标签
#pre = classifier.predict(['samsung','huawei mate 20']) #输出改文本的预测结


#[id_tag_dict.get( v[0].replace('__label__',''))  for v in pre[0]]
#[v[0] for v in pre]
# Test the classifier
#result_test = classifier.test(test_file)
##print(result.precision)
##print(result.recall)
#
#import numpy as np
#result=classifier.predict(data_val['text'].tolist())
#
#data_val['pred']=np.array(result[0])
#data_val['key_pred']=data_val['pred'].str.replace('__label__','')
#
#
#ok=data_val[data_val['key']==data_val['key_pred']]
#bad=data_val[data_val['key']!=data_val['key_pred']]
#
#
#classifier.get_word_vector('phone')
##
#
## Predict some text
## (Example text is from dbpedia.train)
#texts = ['birchas chaim , yeshiva birchas chaim is a orthodox jewish mesivta \
#        high school in lakewood township new jersey . it was founded by rabbi \
#        shmuel zalmen stein in 2001 after his father rabbi chaim stein asked \
#        him to open a branch of telshe yeshiva in lakewood . as of the 2009-10 \
#        school year the school had an enrollment of 76 students and 6 . 6 \
#        classroom teachers ( on a fte basis ) for a student–teacher ratio of \
#        11 . 5 1 .']
#labels = classifier.predict(texts)
#print (labels)