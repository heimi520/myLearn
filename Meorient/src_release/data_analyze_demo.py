# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 09:44:23 2019

@author: Administrator
"""

import pandas as pd
import re
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import keras.layers as layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models  import *
from keras.layers import *
import keras
from keras import *
from keras.models import load_model

dd=pd.read_csv('../data/amazon_data/title_data.csv')
dd=dd[['keyword', 'product']].drop_duplicates()
dd=dd.rename(columns={'keyword':'PRODUCT_TAG_NAME','product':'PRODUCT_NAME'})
dd['PRODUCT_TAG_ID']=dd['PRODUCT_TAG_NAME'].copy()
dd['BUYSELL']='sell'


col=dd['PRODUCT_NAME'].copy()
col=col.str.lower()
col=col.str.replace("'s",'') ##delete 's
col=col.str.replace('[0-9]{1,40}[.][0-9]*','^') ###match decimal

add_punc = '.,;《》？！’ ” “”‘’@#￥% … &×（）——+【】{};；●，。°&～、|:：\n'
punc = punctuation + add_punc
punc=punc.replace('#','').replace('&','').replace('%','').replace('^','').replace(';','')
col=col.apply(lambda x: re.sub(r"[{}]+".format(punc)," ",x))  ##delete dot
col=col.str.replace('([2][0][0-9][0-9])','****') ###match year 20**
col=col.str.replace('\d{1,20}','*')  ##match int number

col=col.str.replace('none','') ##deletel none
col=col.str.replace('(^\s*)|(\s*$)','') ##delete head tail space
stemer = PorterStemmer()

stop_words_dict={v:1 for v in stopwords.words('english')}
stop_words_dict.update({'':1})
stop_words_dict.pop('t')

def filter_stop_and_stem(sentence,stemer,stop_words_dict):
    return ' '.join([stemer.stem(w) for w in sentence.split(' ')])
    
col=col.apply(filter_stop_and_stem,args=(stemer,stop_words_dict,))

dd['text']=col

dd['len']=dd['text'].apply(lambda x:len(x.split(' ')))

aa=dd[['PRODUCT_NAME','text']]

#dd['len'].hist(bins=100)


def get_voc_count(dd):
    tokenizer=Tokenizer(num_words=None, filters='',lower=True,split=' ')  
    tokenizer.fit_on_texts(dd['text'])
    voc_dict=tokenizer.word_docs
    voc_pd=pd.DataFrame( [list(voc_dict.keys()),list(voc_dict.values())],index=['voc','count']).T
    voc_pd=voc_pd.sort_values('count',ascending=False)
    voc_pd.index=range(len(voc_pd))
    voc_pd['rate']=voc_pd['count'].cumsum()/voc_pd['count'].sum()
    return voc_pd

#voc_pd=voc_pd[voc_pd['count']>=10]

d_dict={}
for k, v in enumerate(dd.groupby('PRODUCT_TAG_NAME')):
    name=v[0]
    td=v[1]
    cnt_pd=get_voc_count(td)
    d_dict[name]=cnt_pd
    print(k,name)












