#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:23:38 2019

@author: heimi
"""

      
    
class DetectModelArgs():
    GPU_DEVICES='0' ##-1:CPU        
    SEQ_LEN=200 #100
    MAX_WORDS=200 ##20000 #5000
    VOC_DIM=30 ##100
    BATCH_SIZE=64##64 ###64
    INIT_LEARN_RATE=0.001 ##0.001
    EPOCH_MAX=50
    DROP_OUT_RATE=0.5  ###0.3
    EARLY_STOP_COUNT=6
 
    MODEL_ID='detect_model_base'




