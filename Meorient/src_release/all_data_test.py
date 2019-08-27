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





def cal_count():
    data_his=pd.read_csv('../data/input/data_his.csv')
    data_test=pd.read_csv('../data/input/data_test.csv')
    data_all=pd.concat([data_his,data_test],axis=0)
    cnt_pd=data_all.groupby('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME'].count().to_frame('count')
    return cnt_pd,data_all

cnt_pd,data_all=cal_count()

sd=data_all[data_all['BUYSELL']=='sell']
bd=data_all[data_all['BUYSELL']=='buy']

import matplotlib.pyplot as plt

#sell_count=sd.groupby('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME'].count().sort_values(ascending=False).to_frame('count')
#buy_count=bd.groupby('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME'].count().sort_values(ascending=False).to_frame('count')
#
#sell_count['count'].plot.bar()
#
#buy_count['count'].plot.bar()
#plt.show()
#
#
#bd['len']=bd['PRODUCT_NAME'].apply(lambda x:len(x))
#bd['len'].hist(bins=100)
#
#
#sd['len']=sd['PRODUCT_NAME'].apply(lambda x:len(x))
#sd['len'].hist(bins=100)


#data_test=pd.read_csv('../data/input/data_test.csv')
data_test=data_all.copy()

text2feature=Text2Feature(seq_len=tag_args.SEQ_LEN,is_rename_tag=False,model_id=tag_args.MODEL_ID)
x_test_padded_seqs=text2feature.transform(data_test)

data_test['label']=text2feature.tag2label(data_test['PRODUCT_TAG_ID'])


data_test['text']=text2feature.text_clean(data_test['PRODUCT_NAME'])
aa=data_test[['PRODUCT_NAME','text']]

model=TextCNN(model_id=tag_args.MODEL_ID)
y_pred_class=model.predict_classes(x_test_padded_seqs)
y_pred=text2feature.num2label(y_pred_class)

data_test['pred']=y_pred
data_test['PRODUCT_TAG_NAME']=text2feature.label2tagname(data_test['PRODUCT_TAG_ID'])
data_test['TAG_NAME_PRED']=text2feature.label2tagname(data_test['pred'])

data_test['sample_count']=data_test['PRODUCT_TAG_NAME'].map(cnt_pd['count'].to_dict())

bad=data_test[data_test['pred']!=data_test['label']]
bad=bad[['langfrom','PRODUCT_NAME_ORIG','PRODUCT_NAME','BUYSELL','PRODUCT_TAG_NAME','TAG_NAME_PRED','BUYSELL','sample_count']]

bad.to_csv('../data/output/history_data_check.csv',index=False)

aa= data_all.groupby('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME'].count().sort_values(ascending=False).to_frame('count')




buy_all=data_all[data_all['BUYSELL']=='buy']
sell_all=data_all[data_all['BUYSELL']=='sell']


ss= sell_all.groupby('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME'].count().sort_values(ascending=False).to_frame('count')
bb= buy_all.groupby('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME'].count().sort_values(ascending=False).to_frame('count')



sell_dict={}
for v in data_all.groupby('PRODUCT_TAG_NAME'):
    sell_dict[v[0]]=v[1]
    

buy_dict={}
for v in data_all.groupby('PRODUCT_TAG_NAME'):
    buy_dict[v[0]]=v[1]
    



from sklearn import metrics

def ret_metrics(data_test):
    y_true_class,y_pred_class=data_test['label'],data_test['pred']
    acc=metrics.accuracy_score(y_true_class, y_pred_class)
    f1= metrics.f1_score(y_true_class, y_pred_class, average='weighted')
    print('准确率', acc)
    print('平均f1-score:', f1)
    

data_test['cnt']=1
data_test['ok']=(data_test['label']==data_test['pred']).astype(int)

#ret_metrics(data_test)


ret_pd=data_test.groupby(['PRODUCT_TAG_ID','label']).agg({'cnt':'sum','ok':'sum'}).reset_index() 
ret_pd['acc']=(ret_pd['ok']/ret_pd['cnt']).round(3)

buy_test=data_test[data_test['BUYSELL']=='buy']
sell_test=data_test[data_test['BUYSELL']=='sell']

print('buy metric')

ret_metrics(buy_test)
print('sell metric')
ret_metrics(sell_test)


ret_sell_list=[]
for v in sell_test.groupby('PRODUCT_TAG_NAME'):
    name=v[0]
    td=v[1]
    acc=td['ok'].sum()/len(td)
    ret_sell_list.append([name,acc,len(td)])
    

ret_sell_pd=pd.DataFrame(ret_sell_list,columns=['PRODUCT_TAG_NAME','ACC','SAMPLE_NUM'])





