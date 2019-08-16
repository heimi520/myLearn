#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:29:13 2019

@author: heimi
"""


import pandas as pd
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import multiprocessing 
import pandas as pd
import numpy as np
import nltk
import re
from nltk.stem import PorterStemmer
 
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk import word_tokenize


def default_clean(text):
    '''
    Removes default bad characters
    '''
    if not (pd.isnull(text)):
        # text = filter(lambda x: x in string.printable, text)
        bad_chars = set(["@", "+", '/', "'", '"', '\\','(',')', '', '\\n', '', 
                         '?', '#', ',','.', '[',']', '%', '$', '&', 
                         ';', '!', ';', ':',"*", "_", "=", "}", "{"])
        for char in bad_chars:
            text = text.replace(char, " ")
            text = re.sub('\d+', "", text)
    return text
 
def stop_and_stem(text, stem=True, stemmer = PorterStemmer()):
    '''
    Removes stopwords and does stemming
    '''
    stoplist = stopwords.words('english')
    if stem:
        text_stemmed = [stemmer.stem(word) for word in word_tokenize(text) if word not in stoplist and len(word) >= 3]
    else:
        text_stemmed = [word for word in word_tokenize(text) if word not in stoplist and len(word) >= 3]
    text = ' '.join(text_stemmed)
    return text




md=pd.read_csv('../data/input/data_add_v5.csv')
md=md[md['source']!='add'].rename(columns={'PRODUCT_NAME':'text'})
md['label']=range(len(md))

sample =md[['text', 'label']]
#print ('The shape of the input data frame: {}'.format(sample.shape))


class TaggedDocumentIterator(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield TaggedDocument(words=doc.split(), tags=[self.labels_list[idx]])
 
docLabels = list(sample['label'])
data = list(sample['text'])
sentences = TaggedDocumentIterator(data, docLabels)


model = Doc2Vec(vector_size=100, window=10, min_count=5, workers=11,alpha=0.025, iter=20)
model.build_vocab(sentences)
#model.train(sentences,total_examples=model.corpus_count, epochs=model.iter)
## Store the model to mmap-able files
#model.save('model_docsimilarity.doc2vec')
#

# Load the model
model = Doc2Vec.load('model_docsimilarity.doc2vec')

#Convert the sample document into a list and use the infer_vector method to get a vector representation for it
new_doc_words = 'battery for mobile phone'.split(' ')
new_doc_vec = model.infer_vector(new_doc_words, steps=50, alpha=0.25)
 
#use the most_similar utility to find the most similar documents.
similars = model.docvecs.most_similar(positive=[new_doc_vec])

similars_sentence=sample.iloc[ [v[0] for v in similars],:]['text'].tolist()
for v in similars_sentence:
    print(v)




#
#subdata=md[md['PRODUCT_TAG_NAME']=='All-In-One Printers']
#
#new_doc_words =[[],[] ]# 'battery for mobile phone'.split(' ')
#new_doc_vec = model.infer_vector(new_doc_words, steps=50, alpha=0.25)
#
#
#subdata['vec']=subdata['text'].str.split(' ').apply(lambda x:model.infer_vector(x, steps=50, alpha=0.25))
#
#
#from sklearn.cluster import KMeans
#
#
#
#from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score
#from Bio.Cluster import kcluster
#from Bio.Cluster import clustercentroids
#import matplotlib.pyplot as plt
##%matplotlib inline
#import numpy as np
##data=np.load('/home/philochan/ResExp/genderkernel/1.npy')
#
#data=np.array(subdata['vec'].tolist())
#coef = []
#x=range(2,5)
#for clusters in x:
#    print('clusters///',clusters)
#    clusterid, error, nfound = kcluster(data, clusters, dist='u',npass=100)
#    silhouette_avg = silhouette_score(data, clusterid, metric = 'cosine')
#    coef.append(silhouette_avg)
#  
#e =[i+3 for i,j in enumerate(coef) if j == max(coef)]
#
#plt.plot(x,coef)
#
#
#clusterid, error, nfound = kcluster(data, 2, dist='u',npass=100)
#
#subdata['pred']=clusterid
#
#subdata.groupby('pred')['pred'].count()
#





#test_predict()

# 
#cores = multiprocessing.cpu_count() 
#model = Doc2Vec(
#                documents,
#                dbow_words=1,
#                vector_size=100, 
#                window=5,
#                alpha=0.5,
#                min_alpha=0.001, 
#                min_count=5,
#                max_vocab_size=20000,
##                iter =10,
#                epochs=10,
#                workers=cores,
#                seed=1)


#model.build_vocab(documents)
##model.train(documents)
#model.train(documents,epochs=10,total_words=model.corpus_count)
#
#model.docvecs.most_similar(positive=["dress"])
##Doc2Vec(dbow+w,d200,hs,w8,mc19,t8)

#
#model.most_similar(['dress'])
#model.similar_by_word('')
#
#
#vec = [model.docvecs["men"]]
#model.docvecs.most_similar(vec, topn=11) 



# train(self, data_iterable=None, corpus_file=None, epochs=None, total_examples=None,
#              total_words=None, queue_factor=2, report_delay=1.0, callbacks=(), **kwargs)

#Doc2Vec(dm=0, dbow_words=1, size=200, window=8, min_count=19, iter=10, workers=cores)


#from gensim.test.utils import get_tmpfile
#fname = get_tmpfile("my_doc2vec_model")
# 
#print (fname)
##output: C:\Users\userABC\AppData\Local\Temp\my_doc2vec_model
# 
#load model from saved file
#model.save('doc2vec')
#model2 = Doc2Vec.load('doc2vec')  
## you can continue training with the loaded model!
##If you’re finished training a model (=no more updates, only querying, reduce memory usage), you can do:
# 
#model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
# 
##Infer vector for a new document:
##Here our text paragraph just 2 words
#vector = model.infer_vector(["system", "response"])
#print (vector)
# 


#vector = model.infer_vector(["system", "response"])
#

#model.similar_by_word('sexy')
model.wv.similar_by_word('hospital')













