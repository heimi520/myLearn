#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 23:33:11 2019

@author: heimi
"""

class cnnConfig():
    GPU_DEVICES='0' ##-1:CPU        
    SEQ_LEN=40 #100
    MAX_WORDS=20000 ##20000 #5000
    VOC_DIM=300 ##100
    BATCH_SIZE=64##64 ###64
    INIT_LEARN_RATE=0.0001 ##0.001
    EPOCH_MAX=50
    DROP_OUT_RATE=0.5  ###0.3
    EARLY_STOP_COUNT=6
    IS_STEM=True
    IS_PRE_TRAIN=False
    EMBED_TRAINABLE=True
    Y_NAME_LIST=['output1','output2']
#     MODEL_ID='cnn_model' ###good
#    MODEL_ID='pretrain_cnn_model_test' ###good
#    MODEL_ID='pretrain_cnn_model_test2' ###good
#    MODEL_ID='pretrain_cnn_model_test2_no_pretrain' ###good
    MODEL_ID='pretrain_cnn_model_test2_embed_staic' ###good
    
    
     




