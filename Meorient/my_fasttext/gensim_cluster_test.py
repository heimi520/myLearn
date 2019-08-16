#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 14:29:13 2019

@author: heimi
"""



from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
 
print (common_texts)
 
"""
output:
[['human', 'interface', 'computer'], ['survey', 'user', 'computer', 'system', 'response', 'time'], ['eps', 'user', 'interface', 'system'], ['system', 'human', 'system', 'eps'], ['user', 'response', 'time'], ['trees'], ['graph', 'trees'], ['graph', 'minors', 'trees'], ['graph', 'minors', 'survey']]
"""
 
 
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(common_texts)]
 
print (documents)
"""
output
[TaggedDocument(words=['human', 'interface', 'computer'], tags=[0]), TaggedDocument(words=['survey', 'user', 'computer', 'system', 'response', 'time'], tags=[1]), TaggedDocument(words=['eps', 'user', 'interface', 'system'], tags=[2]), TaggedDocument(words=['system', 'human', 'system', 'eps'], tags=[3]), TaggedDocument(words=['user', 'response', 'time'], tags=[4]), TaggedDocument(words=['trees'], tags=[5]), TaggedDocument(words=['graph', 'trees'], tags=[6]), TaggedDocument(words=['graph', 'minors', 'trees'], tags=[7]), TaggedDocument(words=['graph', 'minors', 'survey'], tags=[8])]
 
"""
 
model = Doc2Vec(documents, size=5, window=2, min_count=1, workers=16)
#Persist a model to disk:
 
from gensim.test.utils import get_tmpfile
fname = get_tmpfile("my_doc2vec_model")
 
print (fname)
#output: C:\Users\userABC\AppData\Local\Temp\my_doc2vec_model
 
#load model from saved file
model.save(fname)
model = Doc2Vec.load(fname)  
# you can continue training with the loaded model!
#If you’re finished training a model (=no more updates, only querying, reduce memory usage), you can do:
 
model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)
 
#Infer vector for a new document:
#Here our text paragraph just 2 words
vector = model.infer_vector(["system", "response"])
print (vector)
 
















