#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 18:04:54 2019

@author: heimi
"""

from nltk.corpus import wordnet as wn 
dog_set = wn.synsets('dog') 
print('dog的同义词集为：', dog_set) 
print('dog的各同义词集包含的单词有：',[dog.lemma_names() for dog in dog_set]) 
print('dog的各同义词集的具体定义是：',[dog.definition() for dog in dog_set]) 
print('dog的各同义词集的例子是：',[dog.examples() for dog in dog_set]) 




