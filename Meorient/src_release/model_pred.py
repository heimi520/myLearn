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
from mylib.model_meorient import *
from DetectModel import *

tag_args=TagModelArgs()
de_args=DetectModelArgs()

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



import pandas as pd

test_dir=r'../data/测试数据 每日核对'
#file_name='6.11 打标样本测试.xlsx'
#file_name='6.12 打标样本测试.xlsx'
file_name='6.13 打标样本测试.xlsx'
test_path=os.path.join(test_dir,file_name)
#
#db_buy=pd.read_excel(test_path,sheet_name=0)[['PRODUCTS']]
#db_sell=pd.read_excel(test_path,sheet_name=1)[['PRODUCTS']]
#db_buy['BUYSELL']='buy'
#db_sell['BUYSELL']='sell'
#data_test=pd.concat([db_buy,db_sell],axis=0)
#data_test=data_test.rename(columns={'PRODUCTS':'PRODUCT_NAME'})

data_test=pd.read_excel(test_path,sheet_name=0).rename(columns={'products':'PRODUCT_NAME'})

data_orig=pd.read_excel(test_path,sheet_name=0).rename(columns={'products':'PRODUCT_NAME'})


#data_test=pd.read_csv('../data/input/data_test.csv')
#data_test=pd.read_csv('../data/input/data_his.csv')




#
#detectpipline=DetectPipLine(seq_len=de_args.SEQ_LEN,is_rename_tag=False,model_id=de_args.MODEL_ID)
#lang_pred=detectpipline.predict(data_test[['PRODUCT_NAME']])
#
#data_test['lang_pred']=lang_pred
#
#en_data=data_test[data_test['lang_pred']=='en']
#other_data=data_test[data_test['lang_pred']!='en']
#
#fromlang='es'
#
#sentence='---'.join(other_data.loc[other_data['lang_pred']==fromlang, 'PRODUCT_NAME'].tolist())

#




js=MyGoogleTransTools()

tolang_list=['ar','es','pl','ru','tr']

#
#fromlang='en'
#tolang='ar'
#
#subdata=dd[['PRODUCT_NAME','PRODUCT_TAG_NAME','PRODUCT_TAG_ID','COUNTRY_NAME','COUNTRY_ID']].copy()
#subdata.index=range(len(subdata))
############################################################
#ret_trans_pd=subdata.copy()
#ret_trans_pd['trans_text']=subdata['PRODUCT_NAME'].copy()
#ret_trans_pd['source_text']=ret_trans_pd['trans_text'].copy()
#ret_trans_pd['idx']=range(len(ret_trans_pd))
#ret_trans_pd['langfrom']='en'
#ret_trans_pd['langto']='en'
#ret_trans_pd=ret_trans_pd[['source_text', 'idx', 'trans_text', 'langfrom', 'langto',
#       'PRODUCT_TAG_NAME', 'PRODUCT_TAG_ID']]
##ret_trans_pd.to_csv('../data/input_lang/en_trans_%s_data.csv'%'en',index=False)
##

small_data=data_test.copy()
small_data['lang_pred']='other'

import time
from MyGoogleTrans import *

js=MyGoogleTransTools()
##sentence='good'
##sentence='Россия только что признала независимость Абхазии и Южной Осетии.,ru'
##sentence='telefono, reloj'
#text=js.translate(sentence,fromLang = fromlang, toLang = 'en')





t0=time.time()
ret_list=[]
for k, v in enumerate(small_data[['lang_pred','PRODUCT_NAME']].values):
    lang=v[0]
    line=v[1]

    t1=time.time()
    text=js.translate(line)
    ret_list.append([lang,line,text])
#    res=js.translate(line)
    t2=time.time()
    time.sleep(0.5)
    print('*************************************************')
    print('lang/////////',lang)
    print('source    ///',line)
    print('dest text //',text)
    print('k//',k,'takes time//',t2-t1,'time longs ///',t2-t0)



ret_pd=pd.DataFrame(ret_list,columns=['lang_pred','PRODUCT_NAME','PRODUCT_NAME_TRANS'])
idx_bad=ret_pd['PRODUCT_NAME_TRANS'].apply(lambda x:len(x))==0
ret_pd.loc[idx_bad,'PRODUCT_NAME_TRANS']=ret_pd.loc[idx_bad,'PRODUCT_NAME'].copy()


data_test=ret_pd.rename(columns={'PRODUCT_NAME':'PRODUCT_NAME_ORIG','PRODUCT_NAME_TRANS':'PRODUCT_NAME'})


text2feature=Text2Feature(seq_len=tag_args.SEQ_LEN,is_rename_tag=True,model_id=tag_args.MODEL_ID)
x_test_padded_seqs=text2feature.transform(data_test)


data_test['text']=text2feature.text_clean(data_test['PRODUCT_NAME'])
aa=data_test[['PRODUCT_NAME','text']]

model=TextCNN(model_id=tag_args.MODEL_ID)
y_pred_class=model.predict_classes(x_test_padded_seqs)
y_pred=text2feature.num2label(y_pred_class)

data_test['pred']=y_pred
data_test['TAG_NAME_PRED']=text2feature.label2tagname(data_test['pred'])

#
#data_test[['PRODUCT_NAME_ORIG','PRODUCT_NAME',  'pred',\
#           'TAG_NAME_PRED']].to_csv(('../data/output/%s_result_check_%s.csv'%(file_name\
#           ,pd.datetime.now().strftime('%Y%m%d'))).replace('.xlsx',''),index=False)

data_test[['PRODUCT_NAME_ORIG','PRODUCT_NAME',  'pred',\
           'TAG_NAME_PRED']].to_csv(('../data/output/%s_result_check_%s.csv'%(file_name\
           ,pd.datetime.now().strftime('%Y%m%d'))).replace('.xlsx',''),index=False)

aa=pd.merge(data_test,data_orig,left_index=True,right_index=True,how='left')





