#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:26:46 2019

@author: heimi
"""

import tkinter as tk




import pandas as pd
import numpy as np
import Levenshtein


def clean(line):
    line=line.lower().replace(' ','')
    return line
#dd=pd.read_excel('标签汇总-new.xlsx')
#dd[['TAG']].to_csv('tag.csv',index=False)

tag=pd.read_csv('tag.csv')
tag['tag_list']=tag['TAG'].apply(lambda x:x.split(' '))

line_list=[]
for tag_name,tag_list in tag[['TAG','tag_list']].values:
    for word in tag_list:
        line_list.append([word,tag_name])  
line_pd=pd.DataFrame(line_list,columns=['word','TAG'])
orig_pd=tag[['TAG']]
orig_pd['word']=orig_pd['TAG']
line_pd=pd.concat([orig_pd,line_pd],axis=0)

tag=line_pd.copy()
tag['dest_clean']=tag['word'].apply(clean)
tag['tag_clean']=tag['TAG'].apply(clean)
tag['len']=tag['tag_clean'].apply(lambda x:len(x))

def check(tag,src_line):
    import time
    t1=time.time()
    tag_list=tag['dest_clean'].tolist()
    dist_list=[]
    src=clean(src_line)
    len_src=len(src)
    for dest_line in  tag_list:
        dist=Levenshtein.distance(src,dest_line[:len_src])
        dist_list.append(dist)
    tag['src']=src_line
    tag['dist']=dist_list 

    tag=tag[tag['dist']<=0]
    
    tag=tag.sort_values('dist')
    sort_list=[]
    dist_list=list(set(tag['dist']))
    for dist in dist_list:
        td=tag[tag['dist']==dist]
        td=td.sort_values('len')
        sort_list.append(td)
    
    show_list=[]
    if len(sort_list)>0:
        sort_pd=pd.concat(sort_list,axis=0)  
        show_list=sort_pd.head(100)['TAG'].unique().tolist()
    t2=time.time()
    print('takes time',t2-t1)
    return show_list










def check_spell(e):
    global text
    var = entry.get()		#调用get()方法，将Entry中的内容获取出来
    
#    check_line=check(tag,var)
    check_line=check_idf(var)
    text.delete('1.0','end')
    for line in check_line:
        text.insert(tk.END, line+'\n')
#    print(var)


import keras.layers as layers
from keras.preprocessing.text import Tokenizer
import re
from string import punctuation
from nltk.stem import PorterStemmer
import random
from nltk.corpus import stopwords
stop_words_dict={v:1 for v in stopwords.words('english')}

from gensim import corpora, models, similarities
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
 


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
    def __init__(self,nlist=10,k=30,):
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


print('idx to sentence //////////////////////')

#gs_int_num_dict=tag.loc[:,'number'].to_dict()
#
#def idx2np(I,D,int_num_dict,int_str_dict):
def idx2np(I,D,int_tag_dict):
    data_list=[]
#    num_list=[]
    for idx_r,line in enumerate(I):
        temp_list=[]
#        tmp_list=[]
        for idx_c,v in enumerate(line):
            if v>=0:
                sentence_values=int_tag_dict[v]
#                num_values=int_num_dict[v]
            else:
                sentence_values=''
#                num_values=0
            temp_list.append(sentence_values)
#            tmp_list.append(num_values)
        data_list.append(temp_list)
#        num_list.append(tmp_list)
#    return data_list,num_list
    return data_list


#########
############
import time
############

tag_uc=tag[['dest_clean']].drop_duplicates()

ftproc=FeatureProc()
ft_train=ftproc.fit_transform(tag_uc['dest_clean'])

simproc=simProc(nlist=10,k=100)
simproc.fit(ft_train)


def check_idf(src_line):
#    global tag
#    global tag_uc
#    
#    src_line='mobile case'
    ft_pred=ftproc.tranform([src_line])
    D,I=simproc.transform(ft_pred)
    
    limit_thresh=0.80
    idx_bad=D<limit_thresh
    D[idx_bad]=-1
    I[idx_bad]=-1
    
    int_tag_dict={k:v for k,v in enumerate(tag_uc['dest_clean'])}
    data_list=idx2np(I,D,int_tag_dict)
    
    name_pd=pd.DataFrame(data_list)
    
    flt=name_pd.T
    flt=flt[flt.iloc[:,0]!='']
    flt.columns=['dest_clean']
    
    res=pd.merge(tag,flt,on=['dest_clean'],how='inner')
    
    res=res.sort_values('len')
    show_list=res['TAG'].unique().tolist()
    return show_list




#check_idf(['mobile'],tag_uc)


#limit_thresh=0.99
#idx_bad=D<limit_thresh
#D[idx_bad]=-1
#I[idx_bad]=-1




window = tk.Tk()
#frame = tk.Frame(window, width=100, height=500)
entry = tk.Entry(window,bd=2,width=100)
entry.bind("<KeyRelease>", check_spell)
entry.pack()

text = tk.Text(window, height=50, width=50)
text.pack()

window.mainloop()






