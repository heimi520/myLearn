# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:44:53 2019

@author: Administrator
"""

import time
import random
import multiprocessing
from multiprocessing import *
import pandas as pd

import re
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

import time


import warnings
warnings.filterwarnings('ignore')
import sys
from os import path
sys.path.append( path.dirname(path.dirname(path.abspath(__file__))))



def filter_stop_and_stem_batch(stemer,k,line_list, return_dict):
    '''worker function'''
    stop_words_dict={v:1 for v in stopwords.words('english')}
    stop_words_dict.update({'':1})
    stop_words_dict.pop('t')
    
    ret_list=[]
    for sentence in line_list:
        line=' '.join([stemer.stem(w) for w in sentence.split(' ')])
        ret_list.append(line)

    return_dict[k] = ret_list



class multiProcess(Process):
    def __init__(self,data,worker):
        super().__init__()
        self.col=data
        self.worker=worker
#        
    def set_worder(self,worker):
        self.worker=worker    
    def run(self):
        t1=time.time()
        col=self.col
        num=4
        batch_size=int(len(col)/num)
        batch_dict={}
        k=0
        for k in range(num-1):
            line_list=col[k*batch_size:(k+1)*batch_size]
            batch_dict[k]=line_list
        line_list=col[(k+1)*batch_size:]
        batch_dict[k+1]=line_list
        stemer = PorterStemmer()
        
        manager = Manager()
        return_dict = manager.dict()
        jobs = []
    
        for (k,line_list) in batch_dict.items():
            logger.info('start worker//%s'%k)
            p = multiprocessing.Process(target=self.worker, args=(stemer,k,line_list,return_dict))
            jobs.append(p)
            p.start()
            
        for k,proc in enumerate(jobs):
            logger.info('join result// %s'%k)
            proc.join()
        
        ret_list=[]
        for k in range(num):
            logger.info('k///%s /// len// %s'%(k,len(return_dict[k])))
            ret_list.extend(return_dict[k])
        t2=time.time()
        logger.info('takes time //%s '%(t2-t1))
        return ret_list
            


#def main():
#    tag_args=TagModelArgs()
#    
#    dd=pd.read_csv('../data/input/text_temp.csv')['text'].tolist()[:10000]     
#    
#    text2feature=Text2Feature(seq_len=tag_args.SEQ_LEN,max_words=tag_args.MAX_WORDS, is_rename_tag=False,is_add_data=False,model_id=tag_args.MODEL_ID)
#    
#    p1=multiProcess(dd,worker=text2feature.filter_stop_and_stem_batch)
#    p1.run()
    

if __name__ == '__main__':
    from model_config import *
    from mylib.model_meorient import *
      
    tag_args=TagModelArgs()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=tag_args.GPU_DEVICES # 使用编号为1，2号的GPU 
    if tag_args.GPU_DEVICES=='-1':
        logger.info('using cpu')
    else:
        logger.info('using gpu:%s'%tag_args.GPU_DEVICES)
    
    
    import keras.backend.tensorflow_backend as KTF 
    gpu_config=tf.ConfigProto()
    gpu_config.gpu_options.per_process_gpu_memory_fraction=0.9 ##gpu memory up limit
    gpu_config.gpu_options.allow_growth = True # 设置 GPU 按需增长,不易崩掉
    session = tf.Session(config=gpu_config) 
    # 设置session 
    KTF.set_session(session)
    

    print('data text clean....')
    data_his=pd.read_csv('../data/input/amazon_his.csv')
    
#    data_his=data_his.head(1000)
    
    data_train,data_val= train_test_split(data_his,test_size=0.05)
    
    text2feature=Text2Feature(seq_len=tag_args.SEQ_LEN,max_words=tag_args.MAX_WORDS, is_rename_tag=False,is_add_data=False,model_id=tag_args.MODEL_ID)
    
    
    
#    dd=pd.read_csv('../data/input/text_temp.csv')['text'].tolist()[:10000]     
    
    text2feature=Text2Feature(seq_len=tag_args.SEQ_LEN,max_words=tag_args.MAX_WORDS, is_rename_tag=False,is_add_data=False,model_id=tag_args.MODEL_ID)
    
    data=text2feature.pre_process(data_train,data_val)
    data['text']= text2feature.noise_clean(data['PRODUCT_NAME'])
    
    myprocess=multiProcess(data['text'].tolist(),worker=filter_stop_and_stem_batch)
    data['text']=myprocess.run()


    x_train_padded_seqs,x_val_padded_seqs,y_train,y_val= text2feature.fit_transform(data)
        
    model=TextCNN(seq_len=tag_args.SEQ_LEN,max_words=tag_args.MAX_WORDS, vocab_len=len(text2feature.vocab),voc_dim=tag_args.VOC_DIM,\
                  num_classes=text2feature.num_classes,is_pre_train=False,\
                  init_learn_rate=tag_args.INIT_LEARN_RATE,batch_size=tag_args.BATCH_SIZE,epoch_max=tag_args.EPOCH_MAX,\
                  drop_out_rate=tag_args.DROP_OUT_RATE,early_stop_count=tag_args.EARLY_STOP_COUNT,model_id=tag_args.MODEL_ID)
    #
    model.build_model()
    model.train(x_train_padded_seqs, y_train,x_val_padded_seqs, y_val)
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
