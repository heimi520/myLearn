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
from mylib.model_meorient2 import *


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

from model_config import *
from mylib.model_meorient2 import *



data_test=all_data

def pipeline_predict(line_list):
    col=pd.DataFrame(line_list,columns=['PRODUCT_NAME'])  
    
    text2feature=Text2Feature(seq_len=tag_args.SEQ_LEN,max_words=tag_args.MAX_WORDS,is_rename_tag=False,model_id=tag_args.MODEL_ID)

    x_test_padded_seqs=text2feature.pipeline_transform(col)
    
    model=TextCNN(max_words=tag_args.MAX_WORDS, model_id=tag_args.MODEL_ID)
    
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
#

    return tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second


#
#
#line_list=['mobile']
##line_list=[' samsung']
##line_list=[' iphone']
line_list=['Cellphones']

#
#line_list=['iphone']
#
#line=line_list[0]
#t=re.sub('[0-9]{1,40}[,][0-9]*','*',line)
#
#line_list=[t]

text2feature=Text2Feature(seq_len=tag_args.SEQ_LEN,max_words=tag_args.MAX_WORDS,is_rename_tag=False,model_id=tag_args.MODEL_ID)

text2feature.text_clean(line_list)
        
#t=re.sub('sma[a-zA-Z]{1,5}watch','smart watch',line_list[0].lower())
#
#line_list=[t]
#print(t)


#line_list=[' White Headphones  ']  ###notice
#line_list=[' telephones ']
#line_list=['Havit i96 TWS /True Wireless Sports IPX6 Waterproof Earphone'] ####notice

#line_list=['Sport Blue Tooth Earphone']  ##notice


#line_list=['  Custom, factory, irregular, crinkled dress in the style of OEM  ']
line_list=['iphone ']

tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second=pipeline_predict(line_list)
print('tag2_max',tag2_max,'prob2_max',prob2_max,'tag1_max',tag1_max )



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


data_test['TAG_NAME_PRED']=data_test['TAG_NAME_PRED1']
#data_test['TAG_NAME_PRED']=np.where(data_test['TAG_prob_pred1']>=0.4,data_test['TAG_NAME_PRED1'],'Other')

data_test2=data_test[['BUYSELL','TAG_NAME_PRED1','TAG_NAME_PRED2','TAG_prob_pred1','T1_NAME_PRED1','T1_prob_pred1','PRODUCT_NAME']]
data_test2.to_csv('../data/output/meorient_tag_pred_mlt_0708_v2.csv',index=False,encoding='utf-8')




##
##aa= data_test.groupby('TAG_NAME_PRED')['TAG_NAME_PRED'].count().sort_values().to_frame('count')
##aa['TAG_NAME_PRED']=aa.index
##
##
##
###cd.to_csv('../data/output/tag_stat_0627.csv')
##
###
##data_test['TAG_NAME_PRED0']=data_test['TAG_NAME_PRED'].copy()
##idx=data_test['prob_max']<0.2
##data_test.loc[idx, 'TAG_NAME_PRED']='other'
#
#data_test['TAG_prob_pred1'].hist(bins=100)


#
#
##text2feature=Text2Feature(seq_len=tag_args.SEQ_LEN,max_words=tag_args.MAX_WORDS,is_rename_tag=False,model_id=tag_args.MODEL_ID)
##x_test_padded_seqs=text2feature.pipeline_transform(data_test)
##
##
##
##model=TextCNN(max_words=tag_args.MAX_WORDS, model_id=tag_args.MODEL_ID)
##
##
##y_pred_class=model.predict_classes(x_test_padded_seqs)
##y_pred=text2feature.num2label(y_pred_class)
##
##data_test['TAG_NAME_PRED']=text2feature.label2tagname(y_pred)
##
#aa=data_test[['PRODUCT_NAME_ORIG','PRODUCT_NAME','TAG_NAME_PRED','TAG_NAME_PRED2', 'prob_max', 'prob_second']]

#pred_dict={}
#for v in aa.groupby('TAG_NAME_PRED'):
#    pred_dict[v[0]]=v[1]




#data_test=pd.read_csv('../data/output/meorient_tag_pred_mlt_0705_v3_check0.csv')
#data_test=pd.read_csv('../data/output/meorient_tag_pred_mlt_0705_v3_check1.csv')
#data_test=pd.read_csv('../data/output/meorient_tag_pred_mlt_0705_v3_check1.csv')

#data_test=pd.read_csv('../data/output/meorient_tag_pred_mlt_0705_adj_v4.csv')
#data_test=pd.read_excel('../data/output/meorient_tag_pred_0705_check.xlsx')
#
#
#tag1=pd.read_excel('../data/ali_data/服装&3C 标签7.02.xlsx',sheet_name=0)
#tag2=pd.read_excel('../data/ali_data/服装&3C 标签7.02.xlsx',sheet_name=1)
#tag=pd.concat([tag1,tag2],axis=0)
#tag['PRODUCT_TAG_NAME']=tag['Product Tag'].str.replace('(^\s*)|(\s*$)','')
#
#tag.to_csv('../data/ali_data/tag_pd.csv')
#
#b=set(data_test['TAG_NAME_PRED1']).intersection(set(tag['PRODUCT_TAG_NAME']))
#    
#c=set(data_test['TAG_NAME_PRED1']) - set(tag['PRODUCT_TAG_NAME'])



#data_filter=pd.merge(data,tag,on=['PRODUCT_TAG_NAME'],how='inner')


cd=data_test.groupby('TAG_NAME_PRED1')['TAG_NAME_PRED1'].count().sort_values(ascending=False).to_frame('count')
cd['tag']=cd.index
cd['pct']=cd['count']/cd['count'].sum()
#cd.to_csv('../data/output/tag_weight.csv',index=False)






cd2=data_test.groupby('T1_NAME_PRED1')['T1_NAME_PRED1'].count().sort_values(ascending=False).to_frame('count')
cd2['tag']=cd2.index
cd2['pct']=cd2['count']/cd2['count'].sum()
#cd.to_csv('../data/output/tag_weight.csv',index=False)




#pred_list=[]
#for v in cd.index:
#    pred_list.append(aa[aa['TAG_NAME_PRED']==v])








