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


cnnconfig=cnnConfig_STEP2()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=cnnconfig.GPU_DEVICES # 使用编号为1，2号的GPU 
if cnnconfig.GPU_DEVICES=='-1':
    logger.info('using cpu')
else:
    logger.info('using gpu:%s'%cnnconfig.GPU_DEVICES)


T_LEVEL_USED='T2' ###

BUYSELL='buy'


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




def read_sell_data():
    data=pd.read_csv('../data/meorient_data/供应商全行业映射标签（20190716修正247url）.csv')
        
    tag_stand=pd.read_excel('../data/tagpack/8.8机器打标目标标签.xlsx')
    tag_stand.columns=['PRODUCT_TAG_NAME','T1','T2']
    
    if T_LEVEL_USED=='T2':
        data=pd.merge(data,tag_stand[['T1','T2']].drop_duplicates(),on=['T1','T2'],how='inner')
        cnt_pd=data.groupby('T2')['T2'].count().to_frame('count')
    elif T_LEVEL_USED=='T1':
        data=pd.merge(data,tag_stand[['T1']].drop_duplicates(),on=['T1'],how='inner')
        cnt_pd=data.groupby('T1')['T1'].count().to_frame('count')
     
    elif T_LEVEL_USED=='T0':
        cnt_pd=data.groupby('T1')['T1'].count().to_frame('count')
        
    return data,tag_stand, cnt_pd





def read_buy_data():
    data_trans=pd.read_csv('../data/tagpack/tagpack1_trans_ret_T1.csv')
    data=pd.read_excel('../data/meorient_data/买家全行业映射标签（20190717）.xlsx',encoding='gbk')
   
    tag_stand=pd.read_excel('../data/tagpack/8.8机器打标目标标签.xlsx',encoding='gbk')
    tag_stand.columns=['PRODUCT_TAG_NAME','T1','T2']
    
    if T_LEVEL_USED=='T2':
         data=pd.merge(data,tag_stand[['T1','T2']].drop_duplicates(),on=['T1','T2'],how='inner')
         cnt_pd=data.groupby('T2')['T2'].count().to_frame('count')
    else:
        data=pd.merge(data,tag_stand[['T1']].drop_duplicates(),on=['T1'],how='inner')
        cnt_pd=data.groupby('T1')['T1'].count().to_frame('count')

    data=pd.merge(data,data_trans[['source_text','trans_text']],left_on=['PRODUCTS_NAME'],right_on=['source_text'],how='inner')
    data=data.rename(columns={'PRODUCTS_NAME':'PRODUCTS_NAME_ORIG','trans_text':'PRODUCT_NAME'})
    return data,tag_stand,cnt_pd





#data,cnt_pd=read_data()

#data['sample_w']=1
#data['source']='meorient_sell'
#cols_list=['PRODUCT_NAME','T2','T1','PRODUCT_TAG_NAME','sample_w', 'source']
#data=data[cols_list]


if BUYSELL=='buy':
    data,tag_stand,cnt_pd=read_buy_data()
elif BUYSELL=='sell':
    data,tag_stand,cnt_pd=read_sell_data()


data_test=data

line_list=['powerbank  ']
tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second=pipeline_predict(line_list)
print('tag_max',tag2_max,'T1 ',tag1_max)


line_list=data_test['PRODUCT_NAME'].tolist()
tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second=pipeline_predict(line_list)

data_test['T1_NAME_PRED']=tag1_max
data_test['TAG_NAME_PRED']=tag2_max

data_test['T2_NAME_PRED']=data_test['TAG_NAME_PRED'].map(tag_stand.set_index('PRODUCT_TAG_NAME')['T2'].to_dict()).fillna('T1_Other')


data_test['cnt']=1
data_test['ok']=(data_test['PRODUCT_TAG_NAME']==data_test['TAG_NAME_PRED']).astype(int)


a=data_test.groupby(['T1_NAME_PRED','TAG_NAME_PRED'])['T1'].count().reset_index()


cols=['PRODUCT_TAG_NAME','TAG_NAME_PRED','T1_NAME_PRED','T2_NAME_PRED','T1','T2','PRODUCT_TAG_NAME','PRODUCT_NAME']

ret_show=data_test[data_test['T1_NAME_PRED']!='Other_T1']

ret_bad=data_test[data_test['T1_NAME_PRED']=='Other_T1']
#ret_show.to_csv('../data/output/tagpack_sell_pred_%s.csv'%T_LEVEL_USED,index=False)
#data_test[cols].to_csv('../data/output/tagpack_meoreint_pred_TAG_step2.csv',index=False)

cols=['TAG_NAME_PRED','T1_NAME_PRED','T2_NAME_PRED','PRODUCT_NAME']
#ret_show.to_excel('../data/output/tagpack_meoreint_pred_TAG_step2.xlsx',index=False,encoding='gbk')
#ret_show.to_excel('../data/output/tagpack_meoreint_pred_TAG_classweight_step2.xlsx',index=False,encoding='gbk')
ret_show.to_excel('../data/output/tagpack_%s_pred_%s.xlsx'%(BUYSELL,T_LEVEL_USED),index=False,encoding='gbk')


aa=data_test.groupby('TAG_NAME_PRED')['T1'].count().sort_values(ascending=False).to_frame('count')
aa['pct']=aa['count']/len(data_test)



bb=ret_show[['TAG_NAME_PRED','T1_NAME_PRED','PRODUCT_NAME']].drop_duplicates()
bb.to_csv('../data/output/tagpack_unique_%s_pred_%s.csv'%(BUYSELL,T_LEVEL_USED),index=False)


#
#
#sm=data_test[['PRODUCT_TAG_NAME','T1','T2','T1_NAME_PRED','T2_NAME_PRED','PRODUCT_NAME']]
#
#
#
#bad=sm[sm['T2_NAME_PRED']!=sm['T2']]
#
#bad[['PRODUCT_TAG_NAME','T2','T2_NAME_PRED','T1','T1_NAME_PRED','PRODUCT_NAME']].to_csv('../data/output/bad_data_tagpack.csv',index=False)
#
#
#bad2=bad[['PRODUCT_NAME','T1','T2','T1_NAME_PRED','T2_NAME_PRED']]
#
#bad_list=[]
#
#bad_dict={}
#from keras.preprocessing.text import Tokenizer
#for v in bad.groupby('T2'):
#    name=v[0]
#    td=v[1]
#    
#    line=td.groupby('T2_NAME_PRED')['T2_NAME_PRED'].count().sort_values(ascending=False).to_frame('count')
#    line['T2']=name
#    line['T2_NAME_PRED']=line.index
##    line=line[line['count']>=5]
#    line=line[['T2','T2_NAME_PRED','count']].sort_values('count',ascending=False)
#    bad_list.append(line)
#    
#    for v in  td.groupby('T2_NAME_PRED'):
#        t2pred_name=v[0]
#        t2=v[1]
#        tokenizer=Tokenizer(num_words=None, filters='',lower=True,split=' ')  
#        tokenizer.fit_on_texts(t2['PRODUCT_NAME'])
#        voc_pd=pd.DataFrame([[k,v] for (k,v) in zip(tokenizer.word_docs.keys(),tokenizer.word_docs.values())] ,columns=['key','count'])
#        voc_pd=voc_pd.sort_values('count',ascending=True)
#        voc_pd['pct']=voc_pd['count'].cumsum()/voc_pd['count'].sum()
#        voc_pd=voc_pd[voc_pd['count']>3]
#        if len(voc_pd)>0:
#            key='[%s]-[%s]-[%s]'%(td['T1'].iloc[0], name,t2pred_name)
#            bad_dict[key]=voc_pd
#            
#    
#    
##    bad_dict[name]=line
#
#aa=data_test[data_test['T1_NAME_PRED']=='Other']
#
#
#bad_pd=pd.concat(bad_list,axis=0)
#
#bad_pd.to_csv('../data/output/tagpack_bad_pred_stat.csv',index=False)
#
#bad_pd=bad_pd.sort_values('count',ascending=False)
#
#
#ret_pd=data_test.groupby('T2').agg({'ok':'sum', 'cnt': 'sum'})
#ret_pd['acc']=ret_pd['ok']/ret_pd['cnt']
#ret_pd['test_smaple_count']=data_test.groupby('T2')['ok'].count()
#ret_pd=pd.merge(ret_pd,cnt_pd,left_index=True,right_index=True,how='left')
#ret_pd=ret_pd.sort_values('acc')
#ret_pd=ret_pd.rename(columns={'count':'train_sample_count'})
#ret_pd['tag']=ret_pd.index
#
##
##
##
##
#
#






