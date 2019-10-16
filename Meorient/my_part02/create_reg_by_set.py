#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 11:09:14 2019

@author: heimi
"""


import pandas as pd
from keras.preprocessing.text import Tokenizer
from nltk.stem import PorterStemmer
import re
from string import punctuation



dd=pd.read_csv('../data/tagpack2/8.19 打标标签my.csv').fillna('')
cols=['PRODUCT_TAG_NAME', 'T1', 'T2', 'trans_text', 'tag1', 'tag2', 'tag3','except']
dd.columns=cols



add_punc = '.,;《》？！’ ” “”‘’@#￥% … &×（）——+【】{};；●，。°&～、|:：\n'
punc = punctuation + add_punc
  
            
stemer=PorterStemmer()
dd['tag_stem']=dd['PRODUCT_TAG_NAME'].apply(lambda x:' '.join([stemer.stem(v) for v in re.sub(r"[{}]+".format(punc)," ",x).lower().split(' ')]) )


ret_list=[]
for v in dd.groupby('T2'):
    td=v[1]
    tokenizer=Tokenizer(num_words=None, filters='',lower=True,split=' ')  
    tokenizer.fit_on_texts(td['tag_stem'])
    t2_tag_set=set(tokenizer.word_index.keys())
    line_list=[]
    for line in  td['tag_stem']:
        line2='  '.join(list(t2_tag_set-set(line.split(' '))))
        line_list.append(line2)
    td['except']=line_list
    ret_list.append(td)

ret_pd=pd.concat(ret_list,axis=0)

ret_pd[cols].to_csv('../data/tagpack2/8.19 打标标签my2.csv',index=False)

    















