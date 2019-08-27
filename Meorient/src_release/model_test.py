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



aa=data_all.groupby('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME'].count().sort_values().to_frame('count')
aa.to_csv('../data/output/data_not_enough.csv')


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
data_test['TAG_NAME_PRED']=tag_max
data_test['TAG_NAME_PRED2']=tag_second
data_test['prob_max']=prob_max
data_test['prob_second']=prob_second


aa= data_test.groupby('TAG_NAME_PRED')['PRODUCT_TAG_NAME'].count().sort_values().to_frame('count')
aa['TAG_NAME_PRED']=aa.index

###########################
#aa=pd.DataFrame(data_all['PRODUCT_TAG_NAME'].unique().tolist(),columns=['tag'])
#
#line_list=aa['tag'].tolist()


#line_list=['tshirt child']


#line_list=['Pedometer Monitor Bracelet Heart Rate Body Fit Fitness Watch Smart Blood Pressure Band Q1']

line_list=['huaweisdfsdaf' ]

tag_max,tag_second,prob_max,prob_second=pipeline_predict(line_list)
print(tag_max,tag_second,prob_max,prob_second)


#
#data_test['prob_max'].hist(bins=100)
#
#data_test['prob_max'].quantile(0.1)

#aa['tag_pred']=ret
#print(ret)
##

#d_dict={}
#for v in data_all.groupby('PRODUCT_TAG_NAME'):
#    d_dict[v[0]]=v[1]
#
#
#
#
#data_test['sample_count']=data_test['PRODUCT_TAG_NAME'].map(cnt_pd['count'].to_dict())
#
bad=data_test[data_test['TAG_NAME_PRED']!=data_test['PRODUCT_TAG_NAME']]

bad2=bad[['PRODUCT_NAME','PRODUCT_TAG_NAME','TAG_NAME_PRED','TAG_NAME_PRED2','prob_max','prob_second']]

bad_list=[]
for v in bad.groupby('PRODUCT_TAG_NAME'):
    name=v[0]
    td=v[1]
    line=td.groupby('TAG_NAME_PRED')['TAG_NAME_PRED'].count().sort_values(ascending=False).to_frame('count')
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
data_test['ok']=(data_test['TAG_NAME_PRED']==data_test['PRODUCT_TAG_NAME']).astype(int)
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







