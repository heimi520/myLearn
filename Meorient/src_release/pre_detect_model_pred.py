#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 23:31:12 2019

@author: heimi
"""


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


#
#data_his=pd.read_csv('../data/input/data_his.csv')
#data_test=pd.read_csv('../data/input/data_test.csv')
#data_test=pd.concat([data_his,data_test],axis=0)
#data_test['T1']='lang'


data_buy=pd.read_csv('../data/meorient_data/buy_data.csv').rename(columns={'PRODUCTS_NAME':'PRODUCT_NAME'})
data_buy['BUYSELL']='buy'

data_sell=pd.read_csv('../data/meorient_data/sell_data.csv')
data_sell['BUYSELL']='sell'
data0=pd.concat([data_buy,data_sell],axis=0)

#data=pd.read_excel('../data/check/7.1部分检查：meorient_tag_pred_0628.xlsx')

data=data0[(data0['T1']=='Apparel')|(data0['T1']=='Consumer Electronics')]
coldata=data[['PRODUCT_NAME']]
coldata=coldata.drop_duplicates()
coldata=coldata[coldata['PRODUCT_NAME'].notnull()]

data_test=coldata
data_test['T1']='lang'




#col=data_test['PRODUCT_NAME']
def pipeline_predict(data_test):
    text2feature=DetectText2Feature(seq_len=de_args.SEQ_LEN,max_words=de_args.MAX_WORDS,is_rename_tag=False,model_id=de_args.MODEL_ID)
    x_test_padded_seqs=text2feature.pipeline_transform(data_test)
       
    model=DetectTextCNN(seq_len=de_args.SEQ_LEN,max_words=de_args.MAX_WORDS,\
                  voc_dim=de_args.VOC_DIM,\
                  num_classes=text2feature.num_classes_list[1],\
                  is_pre_train=False,\
                  init_learn_rate=de_args.INIT_LEARN_RATE,batch_size=de_args.BATCH_SIZE,epoch_max=de_args.EPOCH_MAX,\
                  drop_out_rate=de_args.DROP_OUT_RATE,early_stop_count=de_args.EARLY_STOP_COUNT,model_id=de_args.MODEL_ID)
       
    model.build_model() 
    prob=model.predict(x_test_padded_seqs)
    

    tag_max_int=np.argmax(prob,axis=1)
    [_,tag_max]=text2feature.num2label([tag_max_int,tag_max_int])
    
    prob_max=np.max(prob,axis=1)
    prob_backup=prob.copy()
    for k,v in enumerate(tag_max_int):
        prob_backup[k,v]=0
    
    
    tag_second_int=np.argmax(prob_backup,axis=1)
    [_,tag_second]=text2feature.num2label([tag_second_int,tag_second_int])
    
    prob_second=np.max(prob_backup,axis=1)
    
    return tag_max,tag_second,prob_max,prob_second 



tag_max,tag_second,prob_max,prob_second =pipeline_predict(data_test)
data_test['lang_pred1']=tag_max
data_test['lang_pred2']=tag_max
data_test['prob_max']=prob_max
data_test['prob_second']=prob_second




cols=['PRODUCT_NAME','lang_pred1']

small_data=data_test.loc[data_test['lang_pred1']!='en',cols]
small_data=small_data.rename(columns={'lang_pred1':'fromLang'})
small_data['toLang']='en'
small_data.to_csv('../data/pred_need_trans/small_data.csv',index=False)

en_data=data_test.loc[data_test['lang_pred1']=='en',cols]
en_data=en_data.rename(columns={'PRODUCT_NAME':'source_text'})
en_data['trans_text']=en_data['source_text'].copy()
en_data=en_data.rename(columns={'lang_pred1':'fromLang'})
en_data['toLang']='en'
en_data['batch_idx']=0
en_data['idx']=0

en_data.to_csv('../data/pred_need_trans/en_data.csv',index=False)

#
#import os
#cmd_line="""
#cd /home/heimi/文档/gitWork/myLearn/test_demo/scrapyTrans
#python main.py
#"""
#os.system(cmd_line)
#
#

#
#
#for v in small_data.groupby(['fromLang','toLang']):
#    [fromLang,toLang]=v[0]
#    td=v[1]
#    break

#
#cmd_line="""
#cd /home/heimi/文档/gitWork/myLearn/test_demo/scrapyTrans
#scrapy crawl vsco  -a input_path=/home/heimi/文档/gitWork/myLearn/test_demo/data/pred_need_trans/small_data.csv -a output_path=output_path=/home/heimi/文档/gitWork/myLearn/test_demo/data/pred_need_trans/small_trans_data.csv
#"""
#os.system(cmd_line)




#
##data_test.to_csv('../data/input/pred_data_need_to_trans.csv',index=False)
#
#data_dir='../data/pred_need_trans/'
#for fname in os.listdir(data_dir):
#    path=os.path.join(data_dir,fname)
#    os.remove(path) 
#
#
#
#for v in data_test.groupby('lang_pred1'):
#    td=v[1]
#    td[['PRODUCT_NAME']].to_csv(os.path.join( data_dir, 'from_%s_to_%s.csv'%(v[0],'en') ) ,index=False)
#    
#    























#cnt_pd=data_test.groupby('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME'].count().to_frame('count')
#
##data_test=pd.read_csv('../data/input/lang_test.csv')
#
#text2feature=DetectText2Feature(seq_len=SEQ_LEN,is_rename_tag=True,model_id=MODEL_ID)
#x_test_padded_seqs=text2feature.transform(data_test)
#
#
#model=DetectTextCNN(model_id=MODEL_ID)
#y_pred_class=model.predict_classes(x_test_padded_seqs)
#y_pred=text2feature.num2label(y_pred_class)
#
#data_test['LANG_PRED']=y_pred
##data_test['PRODUCT_TAG_NAME']=text2feature.label2tagname(data_test['PRODUCT_TAG_ID'])
#
#aa=data_test[['COUNTRY_NAME','PRODUCT_NAME','LANGUAGE','LANG_PRED','BUYSELL']]
#
#bb=aa[aa['LANG_PRED']!='en']
#
#ret_dict={}
#for v in data_test.groupby('LANGUAGE'):
#    lang=v[0]
#    td=v[1]
#    ret_dict[lang]=td
#
#
#
#
#small_dict={}
#for v in data_test.groupby('LANG_PRED'):
#    lang=v[0]
#    td=v[1]
#    small_dict[lang]=td
#
#
#
#
#
#
#
#
#
#







