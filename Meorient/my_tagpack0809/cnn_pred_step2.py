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


T_LEVEL_USED='T1' ###

BUYSELL='sell'


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
#    data=pd.read_csv('../data/meorient_data/供应商全行业映射标签（20190716修正247url）.csv')
#    data=pd.read_excel('../data/meorient_data/供应商映射标签（截止20190916）.xlsx')
#    data=pd.read_excel('../data/meorient_data/供应商打标 .xlsx')
#    data=pd.read_excel('../data/meorient_data/8.19验收数据.xlsx')
#    data=pd.read_excel('../data/meorient_data/付费买家跑模型-sa.xlsx').rename(columns={'product':'PRODUCT_NAME'})
    
#    data=pd.read_excel('../data/meorient_data/买家跑机器1.xlsx')
#    data.columns=['PRODUCT_NAME']
#   
#    data=pd.read_excel('../data/meorient_data/供应商未打标.xlsx',sheetname=3)
    data=pd.read_excel('../data/meorient_data/打标需求10.9.xlsx',sheetname=0).rename(columns={'PRODUCTS_NAME':'PRODUCT_NAME'})
    
    tag_stand=pd.read_excel('../data/tagpack/8.8机器打标目标标签.xlsx')
    tag_stand.columns=['PRODUCT_TAG_NAME','T1','T2']
    
    if T_LEVEL_USED=='T2':
        data=pd.merge(data,tag_stand[['T1','T2']].drop_duplicates(),on=['T1','T2'],how='inner')
        cnt_pd=data.groupby('T2')['T2'].count().to_frame('count')
    elif T_LEVEL_USED=='T1':
        data=pd.merge(data,tag_stand[['T1']].drop_duplicates(),on=['T1'],how='inner')
        cnt_pd=data.groupby('T1')['T1'].count().to_frame('count')
     
    elif T_LEVEL_USED=='T0':
        cnt_pd=None
#        cnt_pd=data.groupby('T1')['T1'].count().to_frame('count')
        
    return data,tag_stand, cnt_pd






def read_buy_data():
#    data_trans=pd.read_csv('../data/tagpack/tagpack1_trans_ret_T1.csv')
#    data=pd.read_excel('../data/meorient_data/买家全行业映射标签（20190717）.xlsx',encoding='gbk')
   
#    data_trans=pd.read_csv('../data/tagpack/tagpack1_trans_ret_T0_new2.csv')
#    data=pd.read_excel('../data/meorient_data/买家无T1 TAG.xlsx',encoding='gbk').rename(columns={'PRODUCT_NAME':'PRODUCTS_NAME'})
    
 
#    data_trans=pd.read_csv('../data/tagpack/预注册买家有T级 无标签_T0_transedok.csv')
#    data=pd.read_excel('../data/meorient_data/预注册买家有T级 无标签.xlsx',encoding='gbk').rename(columns={'PRODUCT_NAME':'PRODUCTS_NAME'})
#    
#   
#    data_trans=pd.read_csv('../data/meorient_data/无法映射买家部分-翻译_transed.csv')
#    data=pd.read_excel('../data/meorient_data/无法映射买家部分-翻译.xlsx',encoding='gbk').rename(columns={'PRODUCT_NAME':'PRODUCTS_NAME'})
#    
#    data_trans=pd.read_csv('../data/meorient_data/付费买家数据翻译后打标_transed.csv')
#    data=pd.read_excel('../data/meorient_data/付费买家数据翻译后打标.xlsx',encoding='gbk').rename(columns={'product':'PRODUCTS_NAME'})
    
#    data_trans=pd.read_csv('../data/meorient_data/巴西自取买家翻译+跑模型.xlsx_transed.csv')
#    data=pd.read_excel('../data/meorient_data/巴西自取买家翻译+跑模型.xlsx',encoding='gbk').rename(columns={'PRODUCT_NAME':'PRODUCTS_NAME'})
#    
    data_trans=pd.read_csv('../data/meorient_data/预注册买家提交标签信息表9.27_transed.csv')
    data=pd.read_csv('../data/meorient_data/预注册买家提交标签信息表9.27.csv')
    data.columns=['PRODUCTS_NAME']
    
    
#    data_trans=pd.read_csv('../data/meorient_data/预注册买家映射标签20190916)_transed.csv')
#    data=pd.read_excel('../data/meorient_data/预注册买家映射标签（20190916）.xlsx',encoding='gbk')
#    
    tag_stand=pd.read_excel('../data/tagpack/8.8机器打标目标标签.xlsx',encoding='gbk')
    tag_stand.columns=['PRODUCT_TAG_NAME','T1','T2']
    
    if T_LEVEL_USED=='T2':
         data=pd.merge(data,tag_stand[['T1','T2']].drop_duplicates(),on=['T1','T2'],how='inner')
         cnt_pd=data.groupby('T2')['T2'].count().to_frame('count')
    elif T_LEVEL_USED=='T1':
        data=pd.merge(data,tag_stand[['T1']].drop_duplicates(),on=['T1'],how='inner')
        cnt_pd=data.groupby('T1')['T1'].count().to_frame('count')
    elif T_LEVEL_USED=='T0':
        cnt_pd=None
#        cnt_pd=data.groupby('T1')['T1'].count().to_frame('count')
           
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


#data_test.to_excel('../data/output/tag_sell_T0_pred_apparel_3c.xlsx',encoding='gbk',index=False)
#data_test.to_excel('../data/output/预注册买家有T级 无标签_T2TAG_pred_%s.xlsx'%T_LEVEL_USED,encoding='gbk',index=False)

#data_test.to_excel('../data/output/8.19验收数据_tagpack1_%s.xlsx'%T_LEVEL_USED,encoding='gbk',index=False)
#data_test[data_test['TAG_NAME_PRED']!='Other_TAG'].to_excel('../data/output/0809_无法映射买家部分-翻译_buy_%s.xlsx'%T_LEVEL_USED,encoding='gbk',index=False)
#data_test[data_test['TAG_NAME_PRED']!='Other_TAG'].to_excel('../data/output/0809_付费买家数据翻译后打标_buy_%s.xlsx'%T_LEVEL_USED,encoding='gbk',index=False)

#data_test[data_test['TAG_NAME_PRED']!='Other_TAG'].to_excel('../data/output/0809_付费买家跑模型-sa_buy_%s_%s.xlsx'%(T_LEVEL_USED,pd.datetime.now().strftime('%Y%m%d')),encoding='gbk',index=False)
#data_test[data_test['TAG_NAME_PRED']!='Other_TAG'].to_excel('../data/output/0809_巴西自取买家翻译+跑模型_buy_%s_%s.xlsx'%(T_LEVEL_USED,pd.datetime.now().strftime('%Y%m%d')),encoding='gbk',index=False)
#data_test[data_test['TAG_NAME_PRED']!='Other_TAG'].to_excel('../data/output/0809_供应商映射标签（截止20190916）_sell_%s_%s.xlsx'%(T_LEVEL_USED,pd.datetime.now().strftime('%Y%m%d')),encoding='gbk',index=False)
#data_test[data_test['TAG_NAME_PRED']!='Other_TAG'].to_excel('../data/output/0809_预注册买家映射标签（20190916）_buy_%s_%s.xlsx'%(T_LEVEL_USED,pd.datetime.now().strftime('%Y%m%d')),encoding='gbk',index=False)
#data_test[data_test['TAG_NAME_PRED']!='Other_TAG'].to_excel('../data/output/0809_买家跑机器1_buy_%s_%s.xlsx'%(T_LEVEL_USED,pd.datetime.now().strftime('%Y%m%d')),encoding='gbk',index=False)
data_test[data_test['TAG_NAME_PRED']!='Other_TAG'].to_excel('../data/output/0809_打标需求10.9_%s_%s.xlsx'%(T_LEVEL_USED,pd.datetime.now().strftime('%Y%m%d')),encoding='gbk',index=False)





#data_test['cnt']=1
#data_test['ok']=(data_test['PRODUCT_TAG_NAME']==data_test['TAG_NAME_PRED']).astype(int)
#

#a=data_test.groupby(['T1_NAME_PRED','TAG_NAME_PRED'])['T1'].count().reset_index()


#cols=['PRODUCT_TAG_NAME','TAG_NAME_PRED','T1_NAME_PRED','T2_NAME_PRED','T1','T2','PRODUCT_TAG_NAME','PRODUCT_NAME']
#
#ret_show=data_test[data_test['T1_NAME_PRED']!='Other_T1']
#
#ret_bad=data_test[data_test['T1_NAME_PRED']=='Other_T1']
##ret_show.to_csv('../data/output/tagpack_sell_pred_%s.csv'%T_LEVEL_USED,index=False)
#data_test[cols].to_csv('../data/output/tagpack_meoreint_pred_TAG_step2.csv',index=False)

cols=['TAG_NAME_PRED','T1_NAME_PRED','T2_NAME_PRED','PRODUCT_NAME']
#ret_show.to_excel('../data/output/tagpack_meoreint_pred_TAG_step2.xlsx',index=False,encoding='gbk')
#ret_show.to_excel('../data/output/tagpack_meoreint_pred_TAG_classweight_step2.xlsx',index=False,encoding='gbk')
#ret_show.to_excel('../data/output/tagpack_%s_pred_%s.xlsx'%(BUYSELL,T_LEVEL_USED),index=False,encoding='gbk')


#aa=data_test.groupby('TAG_NAME_PRED')['T1'].count().sort_values(ascending=False).to_frame('count')
#aa['pct']=aa['count']/len(data_test)



#bb=ret_show[['TAG_NAME_PRED','T1_NAME_PRED','PRODUCT_NAME']].drop_duplicates()
#bb.to_csv('../data/output/tagpack_unique_%s_pred_%s.csv'%(BUYSELL,T_LEVEL_USED),index=False)




