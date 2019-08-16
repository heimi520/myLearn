#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 09:36:52 2019

@author: heimi
"""


class fastextConfig():
    VOC_DIM=100 ##100
    WORD_NGRAMS=2
    INIT_LEARN_RATE=0.01 ##0.001
    EPOCH_MAX=5
    IS_PRE_TRAIN=False
    MODEL_ID='fasttext_model' ###good

    
    
    

class cnnConfig():
    GPU_DEVICES='-1' ##-1:CPU        
    SEQ_LEN=40 #100
    MAX_WORDS=20000 ##20000 #5000
    VOC_DIM=100 ##100
    BATCH_SIZE=128##64 ###64
    INIT_LEARN_RATE=0.001 ##0.001
    EPOCH_MAX=50
    DROP_OUT_RATE=0.5  ###0.3
    EARLY_STOP_COUNT=6
    IS_PRE_TRAIN=False
    IS_STEM=True
    EMBED_TRAINABLE=True
    Y_NAME_LIST=['output1','output2']
#     MODEL_ID='cnn_model' ###good
    MODEL_ID='fasttext_model_test' ###good
    
     

