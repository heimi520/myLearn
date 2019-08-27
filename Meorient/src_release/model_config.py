#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 23:33:11 2019

@author: heimi
"""

#
#class TagModelArgs():
#    GPU_DEVICES='0' ##-1:CPU        
#    SEQ_LEN=40 #100
#    MAX_WORDS=10000 ##20000 #5000
#    VOC_DIM=100 ##100
#    BATCH_SIZE=64##64 ###64
#    INIT_LEARN_RATE=0.001 ##0.001
#    EPOCH_MAX=50
#    DROP_OUT_RATE=0.5  ###0.3
#    EARLY_STOP_COUNT=6
# 
##    MODEL_ID='V7_model_release'
##    MODEL_ID='V7_model_release122'
##    MODEL_ID='V7_model_release122_batch64'
##    MODEL_ID='V7_model_release122_batch64_pretrain'
##    MODEL_ID='V7_model_release122_batch64_nopretrain_lr0.0001'
#    MODEL_ID='V7_model_release122_batch64_nopretrain_lr0.001_datachange' ###good  real version
##    MODEL_ID='V7_model_release123_batch64_nopretrain_lr0.001_datachange' 
##    MODEL_ID='V7_model_release1223_batch64_nopretrain_lr0.001_datachange' ###good
#    MODEL_ID='V7_model_topK' ###good
#    
    


#
#class TagModelArgs():
#    GPU_DEVICES='0' ##-1:CPU        
#    SEQ_LEN=40 #100
#    MAX_WORDS=10000 ##20000 #5000
#    VOC_DIM=100 ##100
#    BATCH_SIZE=64##64 ###64
#    INIT_LEARN_RATE=0.001 ##0.001
#    EPOCH_MAX=50
#    DROP_OUT_RATE=0.5  ###0.3
#    EARLY_STOP_COUNT=6
# 
##    MODEL_ID='V7_model_release'
##    MODEL_ID='V7_model_release122'
##    MODEL_ID='V7_model_release122_batch64'
##    MODEL_ID='V7_model_release122_batch64_pretrain'
##    MODEL_ID='V7_model_release122_batch64_nopretrain_lr0.0001'
##    MODEL_ID='V7_model_release122_batch64_nopretrain_lr0.001_datachange' ###good  real version
##    MODEL_ID='V7_model_release123_batch64_nopretrain_lr0.001_datachange' 
##    MODEL_ID='V7_model_release1223_batch64_nopretrain_lr0.001_datachange' ###good
#    MODEL_ID='V7_model_topK' ###good
#    
#     
    

#class TagModelArgs():
#    GPU_DEVICES='0' ##-1:CPU        
#    SEQ_LEN=40 #100
#    MAX_WORDS=10000 ##20000 #5000
#    VOC_DIM=100 ##100
#    BATCH_SIZE=64##64 ###64
#    INIT_LEARN_RATE=0.0001 ##0.001
#    EPOCH_MAX=50
#    DROP_OUT_RATE=0.5  ###0.3
#    EARLY_STOP_COUNT=6
# 
##    MODEL_ID='V7_model_release'
##    MODEL_ID='V7_model_release122'
##    MODEL_ID='V7_model_release122_batch64'
##    MODEL_ID='V7_model_release122_batch64_pretrain'
##    MODEL_ID='V7_model_release122_batch64_nopretrain_lr0.0001'
##    MODEL_ID='V7_model_release122_batch64_nopretrain_lr0.001_datachange' ###good  real version
##    MODEL_ID='V7_model_release123_batch64_nopretrain_lr0.001_datachange' 
##    MODEL_ID='V7_model_release1223_batch64_nopretrain_lr0.001_datachange' ###good
#    MODEL_ID='V9_model_rcnn' ###good
#    
         
    
#
#
#class TagModelArgs():
#    GPU_DEVICES='0' ##-1:CPU        
#    SEQ_LEN=40 #100
#    MAX_WORDS=10000 ##20000 #5000
#    VOC_DIM=100 ##100
#    BATCH_SIZE=64##64 ###64
#    INIT_LEARN_RATE=0.001 ##0.001
#    EPOCH_MAX=50
#    DROP_OUT_RATE=0.5  ###0.3
#    EARLY_STOP_COUNT=6
# 
##    MODEL_ID='V7_model_release'
##    MODEL_ID='V7_model_release122'
##    MODEL_ID='V7_model_release122_batch64'
##    MODEL_ID='V7_model_release122_batch64_pretrain'
##    MODEL_ID='V7_model_release122_batch64_nopretrain_lr0.0001'
##    MODEL_ID='V7_model_release122_batch64_nopretrain_lr0.001_datachange' ###good  real version
##    MODEL_ID='V7_model_release123_batch64_nopretrain_lr0.001_datachange' 
##    MODEL_ID='V7_model_release1223_batch64_nopretrain_lr0.001_datachange' ###good
##    MODEL_ID='V10_model_cnn_avg_max_pool' ###good
#    MODEL_ID='V10_model_cnn_avg_max_pool123' ###good
#    
#             
#
#
#
#class TagModelArgs():
#    GPU_DEVICES='0' ##-1:CPU        
#    SEQ_LEN=40 #100
#    MAX_WORDS=10000 ##20000 #5000
#    VOC_DIM=100 ##100
#    BATCH_SIZE=64##64 ###64
#    INIT_LEARN_RATE=0.001 ##0.001
#    EPOCH_MAX=50
#    DROP_OUT_RATE=0.5  ###0.3
#    EARLY_STOP_COUNT=6
# 
##    MODEL_ID='V7_model_release'
##    MODEL_ID='V7_model_release122'
##    MODEL_ID='V7_model_release122_batch64'
##    MODEL_ID='V7_model_release122_batch64_pretrain'
##    MODEL_ID='V7_model_release122_batch64_nopretrain_lr0.0001'
##    MODEL_ID='V7_model_release122_batch64_nopretrain_lr0.001_datachange' ###good  real version
##    MODEL_ID='V7_model_release123_batch64_nopretrain_lr0.001_datachange' 
##    MODEL_ID='V7_model_release1223_batch64_nopretrain_lr0.001_datachange' ###good
##    MODEL_ID='V10_model_cnn_avg_max_pool' ###good
#    MODEL_ID='V10_model_rnncnn' ###good
#    
#            
     


#
#
#class TagModelArgs():
#    GPU_DEVICES='0' ##-1:CPU        
#    SEQ_LEN=40 #100
#    MAX_WORDS=10000 ##20000 #5000
#    VOC_DIM=100 ##100
#    BATCH_SIZE=64##64 ###64
#    INIT_LEARN_RATE=0.001 ##0.001
#    EPOCH_MAX=50
#    DROP_OUT_RATE=0.5  ###0.3
#    EARLY_STOP_COUNT=6
# 
##    MODEL_ID='V7_model_release'
##    MODEL_ID='V7_model_release122'
##    MODEL_ID='V7_model_release122_batch64'
##    MODEL_ID='V7_model_release122_batch64_pretrain'
##    MODEL_ID='V7_model_release122_batch64_nopretrain_lr0.0001'
##    MODEL_ID='V7_model_release122_batch64_nopretrain_lr0.001_datachange' ###good  real version
##    MODEL_ID='V7_model_release123_batch64_nopretrain_lr0.001_datachange' 
##    MODEL_ID='V7_model_release1223_batch64_nopretrain_lr0.001_datachange' ###good
##    MODEL_ID='V10_model_cnn_avg_max_pool' ###good
#    MODEL_ID='cnnrnn_model' ###good
#    
  


class TagModelArgs():
    GPU_DEVICES='0' ##-1:CPU        
    SEQ_LEN=40 #100
    MAX_WORDS=10000 ##20000 #5000
    VOC_DIM=100 ##100
    BATCH_SIZE=128##64 ###64
    INIT_LEARN_RATE=0.001 ##0.001
    EPOCH_MAX=50
    DROP_OUT_RATE=0.5  ###0.3
    EARLY_STOP_COUNT=6
 
#    MODEL_ID='V7_model_release'
#    MODEL_ID='V7_model_release122'
#    MODEL_ID='V7_model_release122_batch64'
#    MODEL_ID='V7_model_release122_batch64_pretrain'
#    MODEL_ID='V7_model_release122_batch64_nopretrain_lr0.0001'
#    MODEL_ID='V7_model_release122_batch64_nopretrain_lr0.001_datachange' ###good  real version
#    MODEL_ID='V7_model_release123_batch64_nopretrain_lr0.001_datachange' 
#    MODEL_ID='V7_model_release1223_batch64_nopretrain_lr0.001_datachange' ###good
#    MODEL_ID='V10_model_cnn_avg_max_pool' ###good
#    MODEL_ID='rnn_model' ###good
#    MODEL_ID='bigru_model' ###good
#    MODEL_ID='bigru_model_monitor1_2' ###good
    MODEL_ID='bigru_model_monitor2' ###good
    
     
    
            
    
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










