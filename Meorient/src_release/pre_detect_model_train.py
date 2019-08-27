# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:05:29 2019

@author: Administrator
"""

#watch -n 10 nvidia-smi


import warnings
warnings.filterwarnings('ignore')
import sys
from os import path
sys.path.append( path.dirname(path.dirname(path.abspath(__file__))))


from model_config import *
from mylib.DetectModel import *

import pandas as pd
from sklearn.model_selection import train_test_split


de_args=DetectModelArgs()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=de_args.GPU_DEVICES # 使用编号为1，2号的GPU 
if de_args.GPU_DEVICES=='-1':
    logger.info('using cpu')
else:
    logger.info('using gpu:%s'%de_args.GPU_DEVICES)


import keras.backend.tensorflow_backend as KTF 
gpu_config=tf.ConfigProto()
gpu_config.gpu_options.per_process_gpu_memory_fraction=0.9 ##gpu memory up limit
gpu_config.gpu_options.allow_growth = True # 设置 GPU 按需增长,不易崩掉
session = tf.Session(config=gpu_config) 
# 设置session 
KTF.set_session(session)


print('data text clean....')
#data_his=pd.read_csv('../data/input/lang_his.csv')
#data_his['sample_w']=1
##data_his=data_his.head(10000)
#
import os

tolang_list = ['ar', 'pl', 'ru', 'tr', 'es','pt']

line_list=[]
for v in tolang_list:
    path=os.path.join('../data/input_lang/','%s_trans_scrapy.csv'%v)
    td=pd.read_csv(path)
    line_list.append(td)

line_pd=pd.concat(line_list,axis=0)
trans_pd=line_pd[['trans_text','toLang']]
trans_pd.columns=['PRODUCT_NAME','PRODUCT_TAG_NAME']
en_data=pd.DataFrame(line_pd['source_text'].drop_duplicates().tolist(),columns=['PRODUCT_NAME'])
en_data['PRODUCT_TAG_NAME']='en'
total_data=pd.concat([trans_pd,en_data],axis=0)
total_data['PRODUCT_TAG_ID']=total_data['PRODUCT_TAG_NAME'].copy()
total_data['T1']='lang'
total_data['sample_w']=1
total_data=total_data[total_data['PRODUCT_NAME'].notnull()]

#total_data=pd.read_csv()

data_train,data_val= train_test_split(total_data,test_size=0.05)
logger.info('total_data shape//////%s'%total_data.shape[0])

text2feature=DetectText2Feature(seq_len=de_args.SEQ_LEN,max_words=de_args.MAX_WORDS, is_rename_tag=False,is_add_data=False,model_id=de_args.MODEL_ID)
x_train_padded_seqs,x_val_padded_seqs,y1_train,y1_val,y2_train,y2_val,w_sp_train=text2feature.pipeline_fit_transform(data_train,data_val)

#aa=text2feature.tokenizer.word_docs
#
#aa=pd.DataFrame([aa.keys(),aa.values()]).T
#aa.columns=['word','count']
#aa=aa.sort_values('count',ascending=False)
#aa.index=range(len(aa))



#aa=total_data.head(1000)
#total_data['text']=text2feature.text_clean(total_data['PRODUCT_NAME'])

#
#
#d_dict={}
#for v in data_val.groupby('PRODUCT_TAG_NAME'):
#    d_dict[v[0]]=v[1]

#
#aa=total_data['text'].apply(lambda x:len(x.split(' ')))
#aa.hist(bins=100)

#text2feature.save_embed( text2feature.tokenizer)

model=DetectTextCNN(seq_len=de_args.SEQ_LEN,max_words=de_args.MAX_WORDS,\
              voc_dim=de_args.VOC_DIM,\
              num_classes=text2feature.num_classes_list[1],\
              is_pre_train=False,\
              init_learn_rate=de_args.INIT_LEARN_RATE,batch_size=de_args.BATCH_SIZE,epoch_max=de_args.EPOCH_MAX,\
              drop_out_rate=de_args.DROP_OUT_RATE,early_stop_count=de_args.EARLY_STOP_COUNT,model_id=de_args.MODEL_ID)
    
model.build_model()
model.train(x_train_padded_seqs, y2_train,x_val_padded_seqs, y2_val,w_sp_train=None) ##data_train['sample_w'].values
#

#
#
#













#detectpipline=DetectPipLine(seq_len=de_args.SEQ_LEN,max_words=de_args.MAX_WORDS, voc_dim=de_args.VOC_DIM, \
#                            is_rename_tag=False,model_id=de_args.MODEL_ID,is_pre_train=False,\
#                      init_learn_rate=de_args.INIT_LEARN_RATE,batch_size=de_args.BATCH_SIZE,epoch_max=de_args.EPOCH_MAX,\
#                      drop_out_rate=de_args.DROP_OUT_RATE,early_stop_count=de_args.EARLY_STOP_COUNT)
#
#detectpipline.train(data_his)
#
#













