#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 23:33:11 2019

@author: heimi
"""

class cnnConfig():
    GPU_DEVICES='0' ##-1:CPU        
    SEQ_LEN=40 #100
    MAX_WORDS=10000 ##20000 #5000
    VOC_DIM=100 ##100
    BATCH_SIZE=64##64 ###64
    INIT_LEARN_RATE=0.001 ##0.001
    EPOCH_MAX=50
    DROP_OUT_RATE=0.5  ###0.3
    EARLY_STOP_COUNT=6
    IS_PRE_TRAIN=False
    IS_STEM=True
    Y_NAME_LIST=['output1','output2']
    MODEL_ID='cnn_model' ###good
#    MODEL_ID='cnn_model_test' ###good
#    MODEL_ID='cnn_model_temp' ###test
    
     




