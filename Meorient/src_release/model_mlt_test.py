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
from MyGoogleTrans import *


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

def cal_count():
    data_his=pd.read_csv('../data/input/ali_amazon_his.csv')
    data_his['flag']='train'
    data_test=pd.read_csv('../data/input/ali_amazon_test.csv')
    data_test['flag']='test'
    data_all=pd.concat([data_his,data_test],axis=0)
    data_his,data_test = train_test_split(data_all, test_size=0.05)
    
    cnt_pd=data_all.groupby('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME'].count().to_frame('count')
    return cnt_pd,data_all,data_his,data_test

cnt_pd,data_all,data_his,data_test=cal_count()


train=data_all[data_all['flag']=='train']
test=data_all[data_all['flag']=='test']
amazon=train[train['source']=='amazon']
ali=train[train['source']=='ali']


#data_test=data_all
#data_test=data_all[data_all['flag']=='test']
#data_test=data_test[data_test['sect_idx']==1]

data_test=data_all


tag_pd=pd.read_csv('../data/output/tag_rename_v4.csv')
tag_dict=tag_pd.set_index('PRODUCT_TAG_NAME_ORIG')['PRODUCT_TAG_NAME'].to_dict()


data_test['PRODUCT_TAG_NAME']=data_test['PRODUCT_TAG_NAME'].map(tag_dict)
data_test['PRODUCT_TAG_ID']=data_test['PRODUCT_TAG_NAME'].copy()

data_test=data_test[['PRODUCT_TAG_NAME','PRODUCT_TAG_ID','PRODUCT_NAME','source','BUYSELL']]



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
#line_list=[' dress phone coat place ']
#tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second=pipeline_predict(line_list)
#print('tag1_max',tag1_max,'tag2_max',tag2_max,'prob2_max',prob2_max )
#

line_list=data_test['PRODUCT_NAME'].tolist()
tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second=pipeline_predict(line_list)

data_test['T1_PRED_NAME1']=tag1_max
data_test['T1_PRED_NAME2']=tag1_second
data_test['T1_PRED_PROB1']=prob1_max
data_test['T1_PRED_PROB2']=prob1_second


data_test['TAG_PRED_NAME1']=tag2_max
data_test['TAG_PRED_NAME2']=tag2_second
data_test['TAG_PRED_PROB1']=prob2_max
data_test['TAG_PRED_PROB2']=prob2_second

data_test['TAG_PRED_NAME']=np.where(data_test['TAG_PRED_PROB1']>=0.4,data_test['TAG_PRED_NAME1'],'Other')

data_test2=data_test[['PRODUCT_NAME','TAG_PRED_NAME','TAG_PRED_NAME1','TAG_PRED_NAME2','TAG_PRED_PROB1','T1_PRED_NAME1','T1_PRED_PROB1']]
data_test2.to_csv('../data/output/meorient_tag_pred_0703_v7.csv',index=False,encoding='utf-8')


bad=data_test[data_test['TAG_PRED_NAME']!=data_test['PRODUCT_TAG_NAME']]

bad2=bad[['PRODUCT_NAME','PRODUCT_TAG_NAME','TAG_PRED_NAME','TAG_PRED_NAME2','TAG_PRED_PROB1','TAG_PRED_PROB2']]

bad_list=[]
for v in bad.groupby('PRODUCT_TAG_NAME'):
    name=v[0]
    td=v[1]
    line=td.groupby('TAG_PRED_NAME')['TAG_PRED_NAME'].count().sort_values(ascending=False).to_frame('count')
    line['PRODUCT_TAG_NAME']=name
    line['TAG_NAME_PRED']=line.index
    line=line[line['count']>=5]
    line=line[['PRODUCT_TAG_NAME','TAG_NAME_PRED','count']].sort_values('count',ascending=False)
    bad_list.append(line)
#    bad_dict[name]=line

bad_pd=pd.concat(bad_list,axis=0)

bad_pd.to_csv('../data/output/bad_pred_stat.csv',index=False)

bad_data_list=[]
for (name,pred_name) in bad_pd[['PRODUCT_TAG_NAME','TAG_NAME_PRED']].values:
    line=data_test[data_test['PRODUCT_TAG_NAME']==name]
    line2=line[line['TAG_NAME_PRED']==pred_name]
    bad_data_list.append(line2)
#    break
    

bad_data_pd=pd.concat(bad_data_list,axis=0)
bad_data_pd=bad_data_pd[['PRODUCT_NAME','PRODUCT_TAG_NAME','TAG_NAME_PRED','TAG_NAME_PRED2','prob_max','prob_second']]
bad_data_pd.to_csv('../data/output/data_error_detail.csv',index=False)

#bad_pd.to_csv('../data/output/bad_count_pd_v3.csv',index=False)

data_test['cnt']=1
data_test['ok']=(data_test['TAG_PRED_NAME']==data_test['PRODUCT_TAG_NAME']).astype(int)
#############3


aa=data_test[data_test['PRODUCT_TAG_NAME']=='Batteries']



print('acc///',data_test['ok'].sum()/len(data_test))

ret_pd=data_test.groupby('PRODUCT_TAG_NAME').agg({'ok':'sum', 'cnt': 'sum'})
ret_pd['acc']=ret_pd['ok']/ret_pd['cnt']
ret_pd['test_smaple_count']=data_test.groupby('PRODUCT_TAG_NAME')['ok'].count()
ret_pd=pd.merge(ret_pd,cnt_pd,left_index=True,right_index=True,how='left')
ret_pd=ret_pd.sort_values('acc')
ret_pd=ret_pd.rename(columns={'count':'train_sample_count'})
ret_pd['tag']=ret_pd.index


tag_w=pd.read_csv('../data/output/tag_weight.csv')

aa=pd.merge(tag_w,ret_pd,on=['tag'],how='inner')

#
##
##ret_pd.to_csv('../data/output/acc_stat_pd.csv')
##print('acc///',data_test['ok'].sum()/len(data_test))

#ret_pd.head(10).index.tolist()
name_list=['Thermal Printers',
         'Laptop & PDA Bags & Cases',
         'MP3/MP4 Bags & Cases',
         'Bank Uniforms',
         'Home CD DVD & VCD Players',
         'DSLR Cameras',
         'Video Cameras',
         'Women Downcoats & Jackets',
         'Industrial Computer & Accessories',
         'Children Shirts']

            #name='Home Radios'
#name=name_list[2]
#name='Industrial Computer & Accessories'
name='Mobile Phones'
sub=data_test[data_test['PRODUCT_TAG_NAME']==name]
sub=sub[['PRODUCT_TAG_NAME','TAG_NAME_PRED','PRODUCT_NAME','source']]

subbad=sub.loc[sub['PRODUCT_TAG_NAME']!='TAG_NAME_PRED',['PRODUCT_TAG_NAME','TAG_NAME_PRED']].drop_duplicates()


data_dict={}
for name in ret_pd.index:
#    print(name)
    td=data_test[data_test['PRODUCT_TAG_NAME']==name]
    data_dict[name]=td







