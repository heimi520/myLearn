#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 10:03:29 2019

@author: heimi
"""

 
import sys
import gensim
import sklearn
import numpy as np
import pandas as pd



import warnings
warnings.filterwarnings('ignore')
import sys
from os import path
#sys.path.append( path.dirname(path.dirname(path.abspath(__file__))))
from keras.preprocessing.text import Tokenizer
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

import re
from string import punctuation
from nltk.stem import PorterStemmer
import random
from nltk.corpus import stopwords
stop_words_dict={v:1 for v in stopwords.words('english')}




from gensim.models.doc2vec import Doc2Vec, LabeledSentence
 
TaggededDocument = gensim.models.doc2vec.TaggedDocument
 
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)



def get_datasest(md_list):
#    with open("../data/input/clean.csv", 'r') as cf:
#        docs = cf.readlines()
#        print(len(docs))
    
#    dd=pd.read_csv('../data/input/clean_data.csv')
#    docs=dd['COMPANY_CLEAN'].tolist()
    x_train = []
    #y = np.concatenate(np.ones(len(docs)))
    for i, text in enumerate(md_list):
        word_list = text.split(' ')
        l = len(word_list)
        word_list[l-1] = word_list[l-1].strip()
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)
 
    return x_train
 





def getVecs(model, corpus, vector_size):
    vecs = [np.array(model.docvecs[z.tags[0]].reshape(1, vector_size)) for z in corpus]
    return np.concatenate(vecs)
 
def train(x_train, vector_size=200, epoch_num=50):
    print('start training model.....')
    model_dm = Doc2Vec(x_train,min_count=10, window = 3, vector_size = vector_size, workers=12)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=epoch_num)
    model_dm.save('../data/model/doc2vec')
    print('training model ok//////')
    return model_dm
 
def test():
    model_dm = Doc2Vec.load("model_dm_wangyi")
    test_text = ['《', '舞林', '争霸' '》', '十强' '出炉', '复活', '舞者', '澳门', '踢馆']
    inferred_vector_dm = model_dm.infer_vector(test_text)
    print(inferred_vector_dm)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)
 
 
    return sims
 
#if __name__ == '__main__':
    
dd=pd.read_csv('../data/input/clean_data.csv')


aa=dd.groupby('COMPANY_CLEAN').head(1)

#aa=dd['COMPANY_CLEAN'].unique()

md_list=list(set(dd['COMPANY_CLEAN']))


md=pd.DataFrame(md_list,columns=['COMPANY_CLEAN'])


x_train = get_datasest(md_list)
###########################

tokenizer=Tokenizer(num_words=None, filters='',lower=True,split=' ')  
tokenizer.fit_on_texts(dd['COMPANY_CLEAN'])

voc_pd=pd.DataFrame([[k,v] for (k,v) in zip(tokenizer.word_docs.keys(),tokenizer.word_docs.values())] ,columns=['key','count'])
voc_pd=voc_pd.sort_values('count',ascending=False)
voc_pd['idx']=range(len(voc_pd))    




#voc_pd['count'].plot.bar()



###########################

#model_dm = train(x_train)
# 
#model_dm = Doc2Vec.load('../data/model/doc2vec')
#test_text = ['《', '舞林', '争霸' '》', '十强' '出炉', '复活', '舞者', '澳门', '踢馆']
#test_text=dd['COMPANY_CLEAN'].head(1000).tolist()
#inferred_vector_dm = model_dm.infer_vector(test_text)
#print(inferred_vector_dm)
#sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)
# 
#
#



model_dm = Doc2Vec.load('../data/model/doc2vec')
#new_doc_words =dd['COMPANY_CLEAN'].iloc[0].split(' ')


model_dm.wv.most_similar('LIMIT')

new_doc_words='0NOVAPAX VENUS POLYMERS PRIVATE LIMITED'

def line2vec(line):
#    return model_dm.infer_vector(line.split(' '))
    return  model_dm.infer_vector(line.split(' '), alpha=0.1, min_alpha=0.0001, steps=5)
 



#invec1 = model.infer_vector(line.split(' '), alpha=0.1, min_alpha=0.0001, steps=5)
#invec2 = model.infer_vector(doc_words2, alpha=0.1, min_alpha=0.0001, steps=5)

#sims = model.docvecs.most_similar([vec1])#计算训练模型中与句子1相似的内容
#print (sims)



#line1='20 MICRONS LIMITED'
#line2='20 MICRONS LTD'



import numpy as np


def cos_sim(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a 
    :param vector_b: 向量 b
    :return: sim
    """
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim



line1='20 MICRONS LIMIT'
line2='20 MICRONS '

vec1=line2vec(line1)
vec2=line2vec(line2)

cos_sim(vec1,vec2)

print (model_dm.wv.most_similar('UA'))


#corr = np.inner(vec1,vec2)
#

#
#print('test line///',new_doc_words)
#similars = model_dm.docvecs.most_similar([new_doc_vec],topn=10)

similars = model_dm.docvecs.most_similar([vec1])#计算训练模型中与句子1相似的内容
print('line1//',line1)
print('///////////')
#print (similars)

similars_sentence=dd.iloc[ [v[0] for v in similars],:]['COMPANY_CLEAN'].tolist()
for v in similars_sentence:
    print(v)

  
#def plot_similarity(labels, features, rotation):
#    corr = np.inner(features, features)
#    sns.set(font_scale=1.2)
#    g = sns.heatmap(
#    corr,
#    xticklabels=labels,
#    yticklabels=labels,
#    vmin=0,
#    vmax=1,
#    cmap="YlOrRd")
#    g.set_xticklabels(labels, rotation=rotation)
#    g.set_title("Semantic Textual Similarity")
#
#
#def run_and_plot(session_, input_tensor_, messages_, encoding_tensor):
#  message_embeddings_ = session_.run(
#      encoding_tensor, feed_dict={input_tensor_: messages_})
#  plot_similarity(messages_, message_embeddings_, 90)
#
#
#    
#    
#messages_ = [
#   'cloth',
#    'Apparel',
#    'dress',
#    'coat',
#    'battery for mobile phone',
#    'power bank with battery',
#    'battery for power bank',
#    'power bank',
#    'mobile phone',
#]
#
#
#message_embeddings_ = extract_embeddings(model_path, messages_)
#message_embeddings2=np.array( [v.mean(axis=0) for v in message_embeddings_])
#
#corr = np.inner(message_embeddings2, message_embeddings2)
##corr_pd=pd.DataFrame(corr,columns=messages_,index=messages_)
##
##plot_similarity(messages_, message_embeddings2, 90)
#
#


















