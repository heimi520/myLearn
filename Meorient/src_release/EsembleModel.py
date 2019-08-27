#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 16:44:30 2019

@author: heimi
"""





import warnings
warnings.filterwarnings('ignore')
import sys
from os import path
sys.path.append( path.dirname(path.dirname(path.abspath(__file__))))


from mylib.model_meorient import *
import pandas as pd



class TextRNN(object):
    def __init__(self,seq_len=150,max_words=None, voc_dim=50, is_rename_tag=False,model_id='detect_base',is_pre_train=False,\
                      init_learn_rate=0.001,batch_size=16,epoch_max=50,\
                      drop_out_rate=0.5,early_stop_count=5):
        self.seq_len=seq_len
        self.max_words=max_words
        self.voc_dim=voc_dim
        self.is_rename_tag=is_rename_tag
        self.model_id=model_id
        self.is_pre_train=is_pre_train    
        self.init_learn_rate=init_learn_rate
        self.batch_size=batch_size
        self.epoch_max=epoch_max
        self.drop_out_rate=drop_out_rate
        self.early_stop_count=early_stop_count
        self.model_id=model_id
        
        
    
    def train(self,data_his):
        """
        """
        text2feature=DetectText2Feature(seq_len=self.seq_len,max_words=self.max_words,is_rename_tag=self.is_rename_tag,is_add_data=False,model_id=self.model_id)
        
        col=text2feature.text_clean(data_his['PRODUCT_NAME'])        
        data_his=data_his[col.apply(lambda x:len(x.replace(' ','').replace('-','')))>2]
        data_train,data_val= train_test_split(data_his,test_size=0.05)

        x_train_padded_seqs,x_val_padded_seqs,y_train,y_val,w_sp_train=text2feature.pipeline_fit_transform(data_train,data_val)

        model=DetectTextCNN(seq_len=self.seq_len,max_words=self.max_words, vocab_len=len(text2feature.vocab),voc_dim=self.voc_dim,\
                      num_classes=text2feature.num_classes,is_pre_train=self.is_pre_train,\
                      init_learn_rate=self.init_learn_rate,batch_size=self.batch_size,epoch_max=self.epoch_max,\
                      drop_out_rate=self.drop_out_rate,early_stop_count=self.early_stop_count,model_id=self.model_id)
        
        model.build_model()
        model.train(x_train_padded_seqs, y_train,x_val_padded_seqs, y_val)

    
    def predict(self,data_test):
        """
        """
        text2feature=DetectText2Feature(seq_len=self.seq_len, max_words=self.max_words, is_rename_tag=self.is_rename_tag,model_id=self.model_id)
        x_test_padded_seqs=text2feature.pipeline_transform(data_test)
        
        
        model=DetectTextCNN(model_id=self.model_id)
        y_pred_class=model.predict_classes(x_test_padded_seqs)
        y_pred=text2feature.num2label(y_pred_class)
        return y_pred

















