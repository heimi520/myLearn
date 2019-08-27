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
from mylib.model_meorient7 import *

#from DetectModel import *


import pandas as pd
from sklearn.model_selection import train_test_split


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


print('data text clean....')


def read_orig_data():
    #data=pd.read_excel('../data/meorient_data/买家全行业映射标签（20190717）.xlsx')
    #data.to_csv('../data/meorient_data/buy_data.csv',index=False)
    data_buy=pd.read_csv('../data/meorient_data/buy_data.csv').rename(columns={'PRODUCTS_NAME':'PRODUCT_NAME'})
    data_buy['BUYSELL']='buy'
    
    data_sell=pd.read_csv('../data/meorient_data/sell_data.csv')
    data_sell['BUYSELL']='sell'
    data0=pd.concat([data_buy,data_sell],axis=0)
    
    data=data0[(data0['T1']=='Apparel')|(data0['T1']=='Consumer Electronics')]
    data=data.rename(columns={'PRODUCT_NAME':'PRODUCT_NAME_ORIG'})
    return data




def read_trans_data():
    en_data=pd.read_csv('../data/pred_need_trans/en_data.csv')
    small_data=pd.read_csv('../data/pred_need_trans/small_trans_data.csv')
    
    coldata=pd.concat([en_data,small_data],axis=0)
    coldata=coldata.rename(columns={'source_text':'PRODUCT_NAME_ORIG','trans_text':'PRODUCT_NAME'})

    return coldata

import pandas as pd
from sklearn.model_selection import train_test_split

from model_config import *
from mylib.model_meorient7 import *


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



def pipeline_predict(line_list):
    col=pd.DataFrame(line_list,columns=['PRODUCT_NAME'])  
    
    text2feature=Text2Feature(seq_len=tag_args.SEQ_LEN,max_words=tag_args.MAX_WORDS,is_rename_tag=False,model_id=tag_args.MODEL_ID)

    x_test_padded_seqs=text2feature.pipeline_transform(col)
       
    model=TextCNN(seq_len=tag_args.SEQ_LEN,max_words=tag_args.MAX_WORDS,\
                  voc_dim=tag_args.VOC_DIM,\
                  num_classes1=text2feature.num_classes_list[0], num_classes2=text2feature.num_classes_list[1],tokenizer=text2feature.tokenizer ,\
                  cut_list=text2feature.cut_list,is_pre_train=False,\
                  init_learn_rate=tag_args.INIT_LEARN_RATE,batch_size=tag_args.BATCH_SIZE,epoch_max=tag_args.EPOCH_MAX,\
                  drop_out_rate=tag_args.DROP_OUT_RATE,early_stop_count=tag_args.EARLY_STOP_COUNT,model_id=tag_args.MODEL_ID)
    
    model.build_model()
    
    [prob1,prob2]=model.predict(x_test_padded_seqs)

    tag1_max_int=np.argmax(prob1,axis=1)
    tag2_max_int=np.argmax(prob2,axis=1)

    [tag1_max,tag2_max]=text2feature.num2label([tag1_max_int, tag2_max_int])
    
    prob1_max=np.max(prob1,axis=1)
    prob2_max=np.max(prob2,axis=1)
    
    prob1_backup=prob1.copy()
    for k,v in enumerate(tag1_max_int):
        prob1_backup[k,v]=0
    prob2_backup=prob2.copy()
    for k,v in enumerate(tag2_max_int):
        prob2_backup[k,v]=0
    
    
    tag1_second_int=np.argmax(prob1_backup,axis=1)
    tag2_second_int=np.argmax(prob2_backup,axis=1)
    [tag1_second,tag2_second]=text2feature.num2label([tag1_second_int,tag2_second_int ])
    
    prob1_second=np.max(prob1_backup,axis=1)
    prob2_second=np.max(prob2_backup,axis=1)
    
    return tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second
data=read_orig_data()
coldata=read_trans_data()

data_test=coldata


line_list=data_test['PRODUCT_NAME'].tolist()
tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second=pipeline_predict(line_list)

data_test['T1_NAME_PRED1']=tag1_max
data_test['T1_NAME_PRED2']=tag1_second
data_test['T1_prob_pred1']=prob1_max
data_test['T1_prob_pred2']=prob1_second

data_test['TAG_NAME_PRED1']=tag2_max
data_test['TAG_NAME_PRED2']=tag2_second
data_test['TAG_prob_pred1']=prob2_max
data_test['TAG_prob_pred2']=prob2_second

data_test['TAG_NAME_PRED']=data_test['TAG_NAME_PRED1'].apply(lambda x:'Other' if len(re.findall('Other',x))>0 else x)

data_test2=data_test[['TAG_NAME_PRED','TAG_NAME_PRED1','TAG_NAME_PRED2','TAG_prob_pred1','T1_NAME_PRED1','T1_prob_pred1','PRODUCT_NAME']]
data_test2.to_csv('../data/output/meorient_tag_pred_enlish.csv',index=False,encoding='utf-8')


cd=data_test.groupby('TAG_NAME_PRED')['TAG_NAME_PRED'].count().sort_values(ascending=False).to_frame('count')
cd['tag']=cd.index
cd['pct']=cd['count']/cd['count'].sum()
#cd.to_csv('../data/output/tag_weight.csv',index=False)



cd2=data_test.groupby('T1_NAME_PRED1')['T1_NAME_PRED1'].count().sort_values(ascending=False).to_frame('count')
cd2['tag']=cd2.index
cd2['pct']=cd2['count']/cd2['count'].sum()
#cd.to_csv('../data/output/tag_weight.csv',index=False)


data_ret=pd.merge(data,data_test[['TAG_NAME_PRED','TAG_NAME_PRED1','T1_NAME_PRED1','PRODUCT_NAME','PRODUCT_NAME_ORIG','fromLang']],on=['PRODUCT_NAME_ORIG'],how='left')
data_ret['key']=data_ret['TAG_NAME_PRED1'].apply(lambda x:re.sub('\s+','',str(x).lower()))


tag_select=pd.read_excel('../data/meorient_data/服装3C 标签正确率80%+.xlsx')
tag_select.columns=['PRODUCT_TAG_NAME','PRODUCT_TAG_ID_NEW','acc_his']
tag_select['key']=tag_select['PRODUCT_TAG_NAME'].apply(lambda x:re.sub('\s+','',x.lower()))

data_show=pd.merge(data_ret,tag_select[['PRODUCT_TAG_ID_NEW','key']],on=['key'],how='inner')
#
#cols_list=['BUYSELL', 
#           'PRODUCT_TAG_NAME', 'TAG_NAME_PRED1',
#           'T1_NAME_PRED1','PURCHASER_ID', 'SOURCE_TYPE','COUNTRY_ID', 'PRODUCT_TAG_ID','PRODUCT_TAG_ID_NEW','SUPPLIER_ID', 'T1',
#           'T1_ID', 'TAG_NAME_PRED', 'lang_pred','PRODUCT_NAME']


#
#data_show=data_show[cols_list]
#data_show=data_show.astype(str)
#output_name='output/%s.xlsx'%(tag_name.replace('/',''))
#writer = pd.ExcelWriter('../data/output/data_meorient_select_ok.csv')

#data_show.to_excel('../data/output/data_meorient_select_ok222222222.xlsx',index=False)


