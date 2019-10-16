#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 15:30:16 2019

@author: heimi
"""

import pandas as pd

#dd=pd.read_excel('../data/input/inda_clean_td.xlsx')
#dd.to_csv('../data/input/inda_clean_td.csv')
#
#dd=pd.read_csv('../data/input/inda_clean_td.csv',index_col=0)
#md=dd[[ 'COMPANY_CLEAN']].drop_duplicates()
#md.to_csv('../data/input/inda_clean_set.csv')



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



md=pd.read_csv('../data/input/inda_clean_set.csv')

#md['text']=md['COMPANY_CLEAN'].apply(lambda x:' '.join([v for v in  x.replace(' ','')]) ) 



tokenizer=Tokenizer(num_words=None, filters='',lower=True,split=' ',char_level=True)  
tokenizer.fit_on_texts(md['COMPANY_CLEAN'])

voc_pd=pd.DataFrame([[k,v] for (k,v) in zip(tokenizer.word_docs.keys(),tokenizer.word_docs.values())] ,columns=['key','count'])
voc_pd=voc_pd.sort_values('count',ascending=False)


from gensim import corpora, models, similarities


from sklearn.feature_extraction.text import CountVectorizer


from sklearn.feature_extraction.text import TfidfTransformer
 

vectorizer = CountVectorizer(analyzer='char')

#计算个词语出现的次数
X = vectorizer.fit_transform(md['COMPANY_CLEAN'])
#获取词袋中所有文本关键词
word = vectorizer.get_feature_names()

#类调用
transformer = TfidfTransformer()
#将词频矩阵X统计成TF-IDF值
tfidf = transformer.fit_transform(X)

ft_his=tfidf.toarray()

#aa=ft_his[:10,:]
#
#
#ret=pd.DataFrame(aa.T).corr()




#查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重


#aa=X.toarray()

#line_list=['ESS DEE NUTEK INFINITIES PVT',
#'ESS DEE NUTEK INFINITIES PRIVATE LIMITED']


line_list=['INDITEX TRENT RETAIL PRIVATE ',
           'INDITEX TRENT RETAIL PRIVATE LIMIT']

ft_vec=transformer.transform(vectorizer.transform(line_list)).toarray()

aa=pd.DataFrame(ft_vec.T,columns=['x1','x2'])
aa.corr()

#aa=tfidf.toarray()

import numpy as np

def cos_dst(vector1,vector2):
    ret=np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
    return ret


cos_dst(ft_vec[0],ft_vec[1])
      
      
#sen_list=md['COMPANY_CLEAN'].tolist()


from sklearn.cluster import DBSCAN
#
#clf = DBSCAN(eps=0.95,
#                min_samples=1,



#
#import numpy as np
#d = 64                           # dimension
#nb = 00000                      # database size
#nq = 10000                       # nb of queries
#np.random.seed(1234)             # make reproducible
#xb = np.random.random((nb, d)).astype('float32')
#xb[:, 0] += np.arange(nb) / 1000.
#xq = np.random.random((nq, d)).astype('float32')
#xq[:, 0] += np.arange(nq) / 1000.
#
#
#
#ft_his=ft_his.astype('float32')


d=ft_his.shape[1]
xb=ft_his.astype('float32')
xq=xb[:400000]

#import faiss                   # make faiss available
#index = faiss.IndexFlatL2(d)   # build the index
#print(index.is_trained)
#index.add(xb)                  # add vectors to the index
#print(index.ntotal)
#
#

#
#k = 4                          # we want to see 4 nearest neighbors
#D, I = index.search(xq, k)     # actual search
#print(I[:5])                   # neighbors of the 5 first queries
#print(D[-5:])      
##
##



import faiss
nlist = 1000                       #聚类中心的个数
k = 10
################################################33
quantizer = faiss.IndexFlatL2(d)  # the other index

#index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
#
#       # here we specify METRIC_L2, by default it performs inner-product search
#assert not index.is_trained
#index.train(xb)
#assert index.is_trained
#
#index.add(xb)
#
#                  # add may be a bit slower as well
#D, I = index.search(xq, k)     # actual search
#
#print(I[:5,:])                  # neighbors of the 5 last queries
#
#print(D[:5,:])
#
#
#idx_bad=D<0.95
#D[idx_bad]=-1
#I[idx_bad]=-1
#


########
ngpus = faiss.get_num_gpus()
print("number of GPUs:", ngpus)

cpu_index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
assert not cpu_index.is_trained
cpu_index.train(xb)
assert cpu_index.is_trained

gpu_index = faiss.index_cpu_to_all_gpus(cpu_index) # build the index

gpu_index.add(xb)              # add vectors to the index
print(gpu_index.ntotal)

k = 10                          # we want to see 4 nearest neighbors
D, I = gpu_index.search(xq, k) # actual search


idx_bad=D<0.95
D[idx_bad]=-1
I[idx_bad]=-1
#

print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries




int_str_dict=md.iloc[:,0].to_dict()

v_np=np.empty(I.shape,dtype=str)
data_list=[]
for idx_r,line in enumerate(I):
    temp_list=[]
    for idx_c,v in enumerate(line):
        if v>=0:
            sentence_values=int_str_dict[v]
        else:
            sentence_values=''
        temp_list.append(sentence_values)
    data_list.append(temp_list)
       


dd=pd.DataFrame(data_list)








#index.nprobe = 10              # default nprobe is 1, try a few more
#D, I = index.search(xq, k)
#print(I[-5:])             
#
#














 













































































