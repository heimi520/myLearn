#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 17:50:41 2019

@author: heimi
"""

import numpy as np
import faiss
from faiss import normalize_L2
import time

d = 64                           # dimension
nb = 1000000                 # database size
np.random.seed(1234)             # make reproducible
training_vectors= np.random.random((nb, d)).astype('float32')

#normalize_L2(training_vectors)

nlist = 1000  # 聚类中心的个数
k = 50 #邻居个数
quantizer = faiss.IndexFlatIP(d)  # the other index，需要以其他index作为基础

index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
# by default it performs inner-product search
assert not index.is_trained
index.train(training_vectors)
assert index.is_trained
index.nprobe = 300  # default nprobe is 1, try a few more
index.add(training_vectors)  # add may be a bit slower as well
t1=time.time()
D, I = index.search(training_vectors[:2], k)  # actual search
t2 = time.time()
print('faiss kmeans result times {}'.format(t2-t1))
# print(D[:5])  # neighbors of the 5 first queries
print(I[:5])











