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

from cnn_config import *
from mylib.model_meorient_esemble import *
#from detect_config import *

import pandas as pd
from sklearn.model_selection import train_test_split


cnnconfig=cnnConfig_STEP1()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=cnnconfig.GPU_DEVICES # 使用编号为1，2号的GPU 
if cnnconfig.GPU_DEVICES=='-1':
    logger.info('using cpu')
else:
    logger.info('using gpu:%s'%cnnconfig.GPU_DEVICES)



def pipeline_predict(line_list):
    col=pd.DataFrame(line_list,columns=['PRODUCT_NAME'])  
    
    text2feature=Text2Feature(cnnconfig)

    x_test_padded_seqs=text2feature.pipeline_transform(col)
    
        
    cnnconfig.NUM_TAG_CLASSES=text2feature.num_classes_list[1]
    cnnconfig.TOKENIZER=text2feature.tokenizer 
    cnnconfig.CUT_LIST=text2feature.cut_list
     
    model=TextCNN(cnnconfig)
        
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




def read_data():
    ok_other=pd.read_csv('../data/tagpack0830/data_filter.csv')
    data=ok_other[ok_other['sample_w']==1]
    cnt_pd=data.groupby('T2')['T2'].count().to_frame('count')
    return data,cnt_pd



data,cnt_pd=read_data()
cols_list=['PRODUCT_NAME','T2','T1','PRODUCT_TAG_NAME','sample_w', 'source']
data=data[cols_list]


data_test=data

#line_list=['huawei mate 20  ']
#tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second=pipeline_predict(line_list)
#print('tag_max',tag2_max,'T1 ',tag1_max)


line_list=data_test['PRODUCT_NAME'].tolist()
tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second=pipeline_predict(line_list)

data_test['T1_NAME_PRED']=tag1_max
data_test['TAG_NAME_PRED']=tag2_max

data_test['cnt']=1
data_test['ok']=(data_test['PRODUCT_TAG_NAME']==data_test['TAG_NAME_PRED']).astype(int)


sm=data_test[['PRODUCT_TAG_NAME','T1','T2','T1_NAME_PRED','TAG_NAME_PRED','PRODUCT_NAME']]


bad=sm[(sm['TAG_NAME_PRED']!=sm['PRODUCT_TAG_NAME'])&(sm['TAG_NAME_PRED']!='Other_TAG')]
bad.to_excel('../data/output/0830_bad_check.xlsx',index=False,encoding='gbk')



bad[['T1','T2','PRODUCT_TAG_NAME','TAG_NAME_PRED','PRODUCT_NAME']].to_excel('../data/output/tagpack2_bad_data.xlsx',index=False,encoding='gbk')

#bad[['PRODUCT_TAG_NAME','T2','T2_NAME_PRED','T1','T1_NAME_PRED','PRODUCT_NAME']].to_csv('../data/output/bad_data_tagpack.csv',index=False)


bad2=bad[['PRODUCT_NAME','T1','T2','T1_NAME_PRED','TAG_NAME_PRED']]

bad_list=[]

bad_dict={}
from keras.preprocessing.text import Tokenizer
for v in bad.groupby('PRODUCT_TAG_NAME'):
    name=v[0]
    td=v[1]
    
    line=td.groupby('TAG_NAME_PRED')['TAG_NAME_PRED'].count().sort_values(ascending=False).to_frame('count')
    line['PRODUCT_TAG_NAME']=name
    line['TAG_NAME_PRED']=line.index
    

#    line=line[line['count']>=5]
    line=line.sort_values('count',ascending=False)
    bad_list.append(line)
    
    for v in  td.groupby('TAG_NAME_PRED'):
        t2pred_name=v[0]
        t2=v[1]
        tokenizer=Tokenizer(num_words=None, filters='',lower=True,split=' ')  
        tokenizer.fit_on_texts(t2['PRODUCT_NAME'])
        voc_pd=pd.DataFrame([[k,v] for (k,v) in zip(tokenizer.word_docs.keys(),tokenizer.word_docs.values())] ,columns=['key','count'])
        voc_pd=voc_pd.sort_values('count',ascending=False)
        voc_pd['pct']=voc_pd['count'].cumsum()/voc_pd['count'].sum()
        voc_pd=voc_pd[voc_pd['count']>3]
        if len(voc_pd)>0:
            key='[%s]-[%s]-[%s]'%(td['T1'].iloc[0], name,t2pred_name)
            bad_dict[key]=voc_pd
            
    

#
#aa=data_test[data_test['T1_NAME_PRED']=='Other']
#

#bad_pd=pd.concat(bad_list,axis=0)


#bad_pd=bad_pd.sort_values('count',ascending=False)
#

ret_pd=data_test.groupby('PRODUCT_TAG_NAME').agg({'ok':'sum', 'cnt': 'sum'})
ret_pd['acc']=ret_pd['ok']/ret_pd['cnt']
ret_pd['test_smaple_count']=data_test.groupby('T2')['ok'].count()
ret_pd=pd.merge(ret_pd,cnt_pd,left_index=True,right_index=True,how='left')
ret_pd=ret_pd.sort_values('acc')
ret_pd=ret_pd.rename(columns={'count':'train_sample_count'})
ret_pd['tag']=ret_pd.index

#
#
#
#
#
#






