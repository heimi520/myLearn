#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:21:11 2019

@author: heimi
"""

import numpy as np
d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.


#import faiss                   # make faiss available
#index = faiss.IndexFlatL2(d)   # build the index
#print(index.is_trained)
#index.add(xb)                  # add vectors to the index
#print(index.ntotal)
#
#
#aa=xb[:5]
#
#k = 4                          # we want to see 4 nearest neighbors
#D, I = index.search(xb[:5], k) # sanity check
#print(I)
#print(D)
#D, I = index.search(xq, k)     # actual search
#print(I[:5])                   # neighbors of the 5 first queries
#print(I[-5:])     



import faiss
d=ft_his.shape[1]
nlist = 100                       #聚类中心的个数
k = 4
quantizer = faiss.IndexFlatL2(d)  # the other index
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
       # here we specify METRIC_L2, by default it performs inner-product search
assert not index.is_trained
index.train(xb)
assert index.is_trained

index.add(xb)                  # add may be a bit slower as well
D, I = index.search(xq, k)     # actual search
print(I[-5:])                  # neighbors of the 5 last queries
index.nprobe = 10              # default nprobe is 1, try a few more
D, I = index.search(xq, k)
print(I[-5:])             







