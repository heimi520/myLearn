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

data=pd.read_csv('../data/input/all_data_dp.csv')


#data=pd.read_excel('../data/check/7.1部分检查：meorient_tag_pred_0628.xlsx')

data=data[data['PRODUCT_NAME'].notnull()]
data=data[data['PRODUCT_NAME'].apply(lambda x:len(x.replace(' ','').replace('\n',''))>=3)]

"""dropduplicate"""
data=data.groupby(['PRODUCT_NAME']).head(1).reset_index() 


detectpipline=DetectPipLine(seq_len=de_args.SEQ_LEN,max_words=de_args.MAX_WORDS, voc_dim=de_args.VOC_DIM, \
                            is_rename_tag=False,model_id=de_args.MODEL_ID,is_pre_train=False,\
                      init_learn_rate=de_args.INIT_LEARN_RATE,batch_size=de_args.BATCH_SIZE,epoch_max=de_args.EPOCH_MAX,\
                      drop_out_rate=de_args.DROP_OUT_RATE,early_stop_count=de_args.EARLY_STOP_COUNT)

lang_pred=detectpipline.predict(data)


data['lang_pred']=lang_pred


#data['PRODUCT_TAG_NAME'].unique()

small_data=data[data['lang_pred']!='en']
en_data=data[data['lang_pred']=='en']
en_data['langfrom']='en'
en_data['langto']='en'
en_data['PRODUCT_NAME_ORIG']=en_data['PRODUCT_NAME'].copy()

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

all_data=all_data.rename(columns={'PRODUCT_TAG_NAME':'SYS_TAG'})

all_data.to_csv('../data/input/trans_sample_data.csv',index=False)



import warnings
warnings.filterwarnings('ignore')
import sys
from os import path
sys.path.append( path.dirname(path.dirname(path.abspath(__file__))))


from model_config import *
from mylib.model_meorient import *


import pandas as pd
from sklearn.model_selection import train_test_split
 

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



data_test=all_data

def pipeline_predict(line_list):
    col=pd.DataFrame(line_list,columns=['PRODUCT_NAME'])  
    text2feature=Text2Feature(seq_len=tag_args.SEQ_LEN,max_words=tag_args.MAX_WORDS,is_rename_tag=False,model_id=tag_args.MODEL_ID)
    x_test_padded_seqs=text2feature.pipeline_transform(col)

#    data_test['label']=text2feature.tag2label(data_test['PRODUCT_TAG_ID'])
    
    model=TextCNN(max_words=tag_args.MAX_WORDS, model_id=tag_args.MODEL_ID)

    
#    tag_max_int=model.predict_classes(x_test_padded_seqs)
#    tag_max=text2feature.num2label(tag_max_int)
    
    y_prob=model.predict(x_test_padded_seqs)

    tag_max_int=np.argmax(y_prob,axis=1)
    tag_max=text2feature.num2label(tag_max_int)
    prob_max=np.max(y_prob,axis=1)
    
    y_prob2=y_prob.copy()
    for k,v in enumerate(tag_max_int):
        y_prob2[k,v]=0
    
    
    tag_second_int=np.argmax(y_prob2,axis=1)
    tag_second=text2feature.num2label(tag_second_int)
    
    prob_second=np.max(y_prob2,axis=1)
#

    return tag_max,tag_second,prob_max,prob_second
#
line_list=data_test['PRODUCT_NAME'].tolist()
tag_max,tag_second,prob_max,prob_second=pipeline_predict(line_list)
data_test['TAG_NAME_PRED1']=tag_max
data_test['TAG_NAME_PRED2']=tag_second
data_test['prob_max']=prob_max
data_test['prob_second']=prob_second
data_test['second2max']=data_test['prob_second']/data_test['prob_max']
data_test['TAG_NAME_PRED']=data_test['TAG_NAME_PRED1'].copy()
#data_test['TAG_NAME_PRED']=np.where(data_test['prob_max']>=0.5,data_test['TAG_NAME_PRED1'],'Other')
#data_test['TAG_NAME_PRED']=np.where(  (data_test['TAG_NAME_PRED2']=='Other' )|(data_test['TAG_NAME_PRED1']=='Other') ,'Other',data_test['TAG_NAME_PRED1'])



#data_test['prob_max'].hist(bins=100)

#data_test.to_csv('../data/output/meorient_tag_pred_0628.csv',index=False,encoding='utf-8')
#

#data_test.to_csv('../data/output/meorient_tag_pred_rnn0701.csv',index=False,encoding='utf-8')

#data_test.to_csv('../data/output/meorient_tag_pred_0703.csv',index=False,encoding='utf-8')

data_test.to_csv('../data/output/meorient_tag_pred_0704_v2.csv',index=False,encoding='utf-8')


aa= data_test.groupby('TAG_NAME_PRED')['TAG_NAME_PRED'].count().sort_values().to_frame('count')
aa['TAG_NAME_PRED']=aa.index



#cd.to_csv('../data/output/tag_stat_0627.csv')

#
#data_test['TAG_NAME_PRED0']=data_test['TAG_NAME_PRED'].copy()
#idx=data_test['prob_max']<0.2
#data_test.loc[idx, 'TAG_NAME_PRED']='other'

#data_test['prob_max'].hist(bins=100)



#text2feature=Text2Feature(seq_len=tag_args.SEQ_LEN,max_words=tag_args.MAX_WORDS,is_rename_tag=False,model_id=tag_args.MODEL_ID)
#x_test_padded_seqs=text2feature.pipeline_transform(data_test)
#
#
#
#model=TextCNN(max_words=tag_args.MAX_WORDS, model_id=tag_args.MODEL_ID)
#
#
#y_pred_class=model.predict_classes(x_test_padded_seqs)
#y_pred=text2feature.num2label(y_pred_class)
#
#data_test['TAG_NAME_PRED']=text2feature.label2tagname(y_pred)
#
aa=data_test[['PRODUCT_NAME_ORIG','PRODUCT_NAME','TAG_NAME_PRED','TAG_NAME_PRED2', 'prob_max', 'prob_second']]

pred_dict={}
for v in aa.groupby('TAG_NAME_PRED'):
    pred_dict[v[0]]=v[1]




cd=aa.groupby('TAG_NAME_PRED')['TAG_NAME_PRED'].count().sort_values(ascending=False).to_frame('count')
cd['tag']=cd.index
cd['pct']=cd['count']/cd['count'].sum()
#cd.to_csv('../data/output/tag_weight.csv',index=False)



pred_list=[]
for v in cd.index:
    pred_list.append(aa[aa['TAG_NAME_PRED']==v])








