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


data_dir='../data/output/%s'%tag_args.MODEL_ID
if not os.path.exists(data_dir):
    os.makedirs(data_dir)




def cal_count():
    data_his=pd.read_csv('../data/input/amazon_his.csv')
    data_test=pd.read_csv('../data/input/amazon_test.csv')
    data_all=pd.concat([data_his,data_test],axis=0)
    cnt_pd=data_all.groupby('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME'].count().to_frame('count')
    return cnt_pd,data_all

cnt_pd,data_all=cal_count()
data_test=pd.read_csv('../data/input/amazon_test.csv')

text2feature=Text2Feature(seq_len=tag_args.SEQ_LEN,max_words=tag_args.MAX_WORDS,is_rename_tag=False,model_id=tag_args.MODEL_ID)
x_test_padded_seqs=text2feature.transform(data_test)

data_test['label']=text2feature.tag2label(data_test['PRODUCT_TAG_ID'])

model=TextCNN(max_words=tag_args.MAX_WORDS, model_id=tag_args.MODEL_ID)
y_pred_class=model.predict_classes(x_test_padded_seqs)
y_pred=text2feature.num2label(y_pred_class)

data_test['text']=text2feature.text_clean(data_test['PRODUCT_NAME'])
data_test['pred']=y_pred
data_test['PRODUCT_TAG_NAME']=text2feature.label2tagname(data_test['PRODUCT_TAG_ID'])
data_test['TAG_NAME_PRED']=text2feature.label2tagname(data_test['pred'])

data_test['sample_count']=data_test['PRODUCT_TAG_NAME'].map(cnt_pd['count'].to_dict())

bad=data_test[data_test['pred']!=data_test['label']]

#bad2=bad[['PRODUCT_TAG_NAME','TAG_NAME_PRED']].drop_duplicates()
#set_list=[]
#for v in bad2.values:
#    set_list.append(frozenset(v))
#aa=set(set_list)
#bad_list=[ list(v) for v in aa]
#bad_pd=pd.DataFrame(bad_list,columns=['tag1','tag2']).sort_values('tag1')
#bad_pd.to_excel('../data/output/bad.xlsx')

data_test['cnt']=1
data_test['ok']=(data_test['label']==data_test['pred']).astype(int)

print('acc///',data_test['ok'].sum()/len(data_test))

ret_list=[]
for v in data_test.groupby('PRODUCT_TAG_NAME'):
    name=v[0]
    td=v[1]
    acc=td['ok'].sum()/len(td)
    ret_list.append([name,acc,len(td),td['sample_count'].iloc[0]])
ret_pd=pd.DataFrame(ret_list,columns=['name','acc','count','sample_count'])
ret_pd=ret_pd.sort_values('acc')

#ret_pd.to_csv(os.path.join(data_dir,'test_result.csv'))


#ret_pd['name'].head(10).tolist()

name_list=['Women Coats',
         'Portable DVD& VCD Players',
         'Electronic Cigarettes &Accessories',
         'Men Downcoats',
         'Wedding Dresses',
         'Portable CD Player',
         'TV & Movie Costumes',
         'Men overcoats',
         'MP4 Players',
         'Men Coats']

sub=data_test[data_test['PRODUCT_TAG_NAME']==name_list[0]]
sub=sub[['PRODUCT_TAG_NAME','TAG_NAME_PRED','PRODUCT_NAME','text']]











