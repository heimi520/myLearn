#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 23:33:11 2019

@author: heimi
"""

class cnnConfig_STEP1():
    GPU_DEVICES='-1' ##-1:CPU        
    SEQ_LEN=40 #100
    MAX_WORDS=200000 ##20000 #5000
    VOC_DIM=30 ##100
    BATCH_SIZE=128##64 ###64
    INIT_LEARN_RATE=0.001 ##0.001
    EPOCH_MAX=50
    DROP_OUT_RATE=0.5  ###0.3
    EARLY_STOP_COUNT=6
    IS_PRE_TRAIN=False
    IS_STEM=True
    EMBED_TRAINABLE=True
    IS_USE_CHAR=False
    LABEL_LIST=['T1','PRODUCT_TAG_NAME']
    Y_NAME_LIST=['output1','output2']
    MODEL_ID='cnn_tagpack0830_step1'
#    MODEL_ID='cnn_tagpack0830_step1_temp'


class cnnConfig_STEP2():
    GPU_DEVICES='-1' ##-1:CPU        
    SEQ_LEN=40 #100
    MAX_WORDS=200000 ##20000 #5000
    VOC_DIM=30 ##100
    BATCH_SIZE=128##64 ###64
    INIT_LEARN_RATE=0.001 ##0.001
    EPOCH_MAX=50
    DROP_OUT_RATE=0.5  ###0.3
    EARLY_STOP_COUNT=6
    IS_PRE_TRAIN=False
    IS_STEM=True
    EMBED_TRAINABLE=True
    LABEL_LIST=['T1','PRODUCT_TAG_NAME']
    Y_NAME_LIST=['output1','output2']
    MODEL_ID='cnn_tagpack2_step2'
   
     
     




class cnnConfig_CHAR_STEP1():
    GPU_DEVICES='0' ##-1:CPU        
    SEQ_LEN=100 #100
    MAX_WORDS=200000 ##20000 #5000
    VOC_DIM=32 ##100
    BATCH_SIZE=128##64 ###64
    INIT_LEARN_RATE=0.001 ##0.001
    EPOCH_MAX=50
    DROP_OUT_RATE=0.5  ###0.3
    EARLY_STOP_COUNT=6
    IS_PRE_TRAIN=False
    IS_STEM=True
    EMBED_TRAINABLE=True
    LABEL_LIST=['T1','PRODUCT_TAG_NAME']
    Y_NAME_LIST=['output1','output2']
#    MODEL_ID='cnn_tagpack_release_demo_sample_w_step1'
    MODEL_ID='cnn_tagpack_char_step1'


class cnnConfig_CHAR_STEP2():
    GPU_DEVICES='0' ##-1:CPU        
    SEQ_LEN=40 #100
    MAX_WORDS=200000 ##20000 #5000
    VOC_DIM=16 ##100
    BATCH_SIZE=128##64 ###64
    INIT_LEARN_RATE=0.001 ##0.001
    EPOCH_MAX=50
    DROP_OUT_RATE=0.5  ###0.3
    EARLY_STOP_COUNT=6
    IS_PRE_TRAIN=False
    IS_STEM=True
    EMBED_TRAINABLE=True
    LABEL_LIST=['T1','PRODUCT_TAG_NAME']
    Y_NAME_LIST=['output1','output2']
#    MODEL_ID='cnn_tagpack_release_demo_sample_w_TAG_STEP2'
#    MODEL_ID='cnn_tagpack_release_demo_sample_w_TAG_classweight_STEP2'
    MODEL_ID='cnn_tagpack_char_step2'
   
     



