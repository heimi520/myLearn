#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 11:49:16 2019

@author: heimi
"""



import pandas as pd
import numpy as np
#
#def to_categorical(y, num_classes=None, dtype='float32'):
#    """Converts a class vector (integers) to binary class matrix.
#
#    E.g. for use with categorical_crossentropy.
#
#    # Arguments
#        y: class vector to be converted into a matrix
#            (integers from 0 to num_classes).
#        num_classes: total number of classes.
#        dtype: The data type expected by the input, as a string
#            (`float32`, `float64`, `int32`...)
#
#    # Returns
#        A binary matrix representation of the input. The classes axis
#        is placed last.
#    """
#    y = np.array(y, dtype='int')
#    input_shape = y.shape
#    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
#        input_shape = tuple(input_shape[:-1])
#    y = y.ravel()
#    if not num_classes:
#        num_classes = np.max(y) + 1
#    n = y.shape[0]
##    categorical = np.zeros((n, num_classes), dtype=dtype)
#    categorical = np.array([[0]*num_classes for v in range(n) ])
#    categorical[np.arange(n), y] = 1
#    output_shape = input_shape + (num_classes,)
#    categorical = np.reshape(categorical, output_shape)
#    return categorical
#
#




#col=[1,2,3,2,4,5,2]



#batch_list=to_batch(col,batch_size=2)
#

import keras
#
#def batch_to_categorical(col,num_classes,batch_size=1000000,dtype='float32'):
#    
#    def to_batch(col, batch_size):
#        import math
#        num=math.ceil(len(col)/batch_size)
#        batch_list=[]
#        for v in range(num):
#            batch_list.append(col[v*batch_size:(v+1)*batch_size] )
#        return batch_list
#            

#    
#    batch_list=to_batch(col, batch_size)
#    
#    ret_list=[]
#    for v in batch_list:
#        ret_batch=to_categorical(v, num_classes=None, dtype=dtype)
#        ret_list.append(ret_batch)
#    ret_np=np.row_stack(ret_list)
#    return ret_np
#
##        

import keras
import gc
def batch_to_categorical(col,num_classes,batch_size=1000000,dtype='float32'):
        
    def to_batch(col, batch_size):
        import math
        num=math.ceil(len(col)/batch_size)
        batch_list=[]
        for v in range(num):
            batch_list.append(col[v*batch_size:(v+1)*batch_size] )
        return batch_list
            

    
    batch_list=to_batch(col, batch_size)
    
    ret_list=[]
    for k,v in enumerate(range(len(batch_list))):
        print('batch////',k)
        ret_batch=keras.utils.to_categorical(batch_list[0], num_classes, dtype=dtype)
        ret_list.extend(ret_batch.tolist())
        del ret_batch
        del batch_list[0]
        gc.collect()
#    ret_np=np.row_stack(ret_list)
    return ret_list

        



num_classes=4000
col=list(range(1,num_classes))*100


md_list=batch_to_categorical(col,num_classes,batch_size=100000,dtype='float16')


#aa=np.row_stack(md)


#print('merge...............')
#a_list=[]
#import gc
#for v in range(len(md)):
#    a_list.extend(md[0].tolist())
#    del md[0]
#    gc.collect()
#    
#    
#print('merge ok/////////')   


#
#ret_list=[]
#for v in range(100):
#    ret_list.append(md)
#    
#
#ret_pd=np.column_stack(ret_list)

#aa=np.zeros((1000000, 5000), dtype='float32')



#[0]*100
##np.zeros((n, num_classes), dtype=dtype)
#from scipy.sparse import coo_matrix
#
#aa=coo_matrix((3000000, 4000)).toarray()
#








