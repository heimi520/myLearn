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
from mylib.model_meorient_rnncnn import *

from mylib.DetectModel import *


import pandas as pd
from sklearn.model_selection import train_test_split



class TagModelArgs():
    GPU_DEVICES='0' ##-1:CPU        
    SEQ_LEN=40 #100
    MAX_WORDS=10000 ##20000 #5000
    VOC_DIM=100 ##100
    BATCH_SIZE=64##64 ###64
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
    MODEL_ID='V10_model_rnncnn' ###good
    

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

data=pd.read_csv('../data/input/all_data_dp.csv').rename(columns={'PRODUCT_NAME':'PRODUCT_NAME_ORIG'})


#data=pd.read_excel('../data/check/7.1部分检查：meorient_tag_pred_0628.xlsx')

data=data[data['PRODUCT_NAME_ORIG'].notnull()]
data=data[data['PRODUCT_NAME_ORIG'].apply(lambda x:len(x.replace(' ','').replace('\n',''))>=3)]

"""dropduplicate"""
data=data.groupby(['PRODUCT_NAME_ORIG']).head(1).reset_index() 

tag_max,tag_second,prob_max,prob_second =detect_pipeline_predict(data,de_args)
data['lang_pred']=tag_max

small_data=data[data['lang_pred']!='en']
en_data=data[data['lang_pred']=='en']
en_data['langfrom']='en'
en_data['langto']='en'
en_data['PRODUCT_NAME']=en_data['PRODUCT_NAME_ORIG'].copy()

print('small data shape',small_data.shape)
#


IS_TRANS_LANG=False
import time
#from MyGoogleTrans import *

tolang_list=['ar','es','pl','ru','tr']
    
if IS_TRANS_LANG:
    js=MyGoogleTransTools()
    for fromlang in tolang_list:
        ##########################################################################
        subdata=small_data.loc[small_data['lang_pred']==fromlang, ['PRODUCT_NAME','PRODUCT_TAG_NAME','PRODUCT_TAG_ID','COUNTRY_NAME','COUNTRY_ID']].copy()
        subdata.index=range(len(subdata))
        batch_idx_list, batch_line_list,batch_str_list=to_batch_data(subdata)
        
        batch_trans_pd=trans_batch(js,fromlang,'en',batch_idx_list,batch_line_list,batch_str_list)
        ret_trans_pd=pd.merge(batch_trans_pd,subdata[['PRODUCT_TAG_NAME','PRODUCT_TAG_ID']],left_on=['idx'],right_index=True,how='left')
        ret_trans_pd=ret_trans_pd.rename(columns={'trans_text':'PRODUCT_NAME','source_text':'PRODUCT_NAME_ORIG'})
        ret_trans_pd.to_csv('../data/input_lang/class_train_data_trans_%s.csv'%fromlang,index=False)
      
     

trans_list=[]
d_dict={}
for fromlang in tolang_list:
    ##########################################################################
    subdata=small_data.loc[small_data['lang_pred']==fromlang, ['PRODUCT_NAME','PRODUCT_TAG_NAME','PRODUCT_TAG_ID','COUNTRY_NAME','COUNTRY_ID','BUYSELL']].copy()
    subdata.index=range(len(subdata))
    
    ret_trans_pd=pd.read_csv('../data/input_lang/class_train_data_trans_%s.csv'%fromlang)
    d_dict[fromlang]=ret_trans_pd
    
    ret_trans_pd=pd.merge(ret_trans_pd,subdata[['BUYSELL']],left_index=True,right_index=True,how='left')
    trans_list.append(ret_trans_pd)
    
    
    
trans_pd=pd.concat(trans_list,axis=0)
cols_list=['PRODUCT_NAME_ORIG','PRODUCT_NAME', 'langfrom', 'langto',
       'PRODUCT_TAG_NAME', 'PRODUCT_TAG_ID', 'BUYSELL']

all_data=pd.concat([en_data[cols_list],trans_pd[cols_list]],axis=0)

all_data=all_data[['PRODUCT_NAME_ORIG', 'PRODUCT_NAME', 'langfrom', 'langto',
       'BUYSELL']]


all_data.to_csv('../data/input/trans_sample_data.csv',index=False)



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


from mylib.model_meorient_rnncnn import *
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



data_test=all_data


line_list=['huawei mate 20']
tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second=pipeline_predict(line_list)
print('tagpred',tag1_max)


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
#data_test['TAG_NAME_PRED']=np.where(data_test['TAG_prob_pred1']>=0.4,data_test['TAG_NAME_PRED1'],'Other')

data_test2=data_test[['BUYSELL','TAG_NAME_PRED','TAG_NAME_PRED1','TAG_NAME_PRED2','TAG_prob_pred1','T1_NAME_PRED1','T1_prob_pred1','PRODUCT_NAME']]
data_test2.to_csv('../data/output/meorient_tag_pred_mlt_0712_test1.csv',index=False,encoding='utf-8')


cd=data_test.groupby('TAG_NAME_PRED')['TAG_NAME_PRED'].count().sort_values(ascending=False).to_frame('count')
cd['tag']=cd.index
cd['pct']=cd['count']/cd['count'].sum()
#cd.to_csv('../data/output/tag_weight.csv',index=False)


cd2=data_test.groupby('T1_NAME_PRED1')['T1_NAME_PRED1'].count().sort_values(ascending=False).to_frame('count')
cd2['tag']=cd2.index
cd2['pct']=cd2['count']/cd2['count'].sum()
#cd.to_csv('../data/output/tag_weight.csv',index=False)














