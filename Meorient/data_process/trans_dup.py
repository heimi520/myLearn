#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 09:32:53 2019

@author: heimi
"""



import pandas as pd

T_LEVEL='T0'

data=pd.read_excel('../data/meorient_data/预注册买家映射标签（20190916）.xlsx',encoding='gbk')



tag1=pd.read_excel('../data/ali_data/服装&3C 标签7.02_new.xlsx',sheet_name=0)
tag2=pd.read_excel('../data/ali_data/服装&3C 标签7.02_new.xlsx',sheet_name=1)
tag=pd.concat([tag1,tag2],axis=0)
tag['PRODUCT_TAG_NAME']=tag['Product Tag'].str.replace('(^\s*)|(\s*$)','')
tag=tag[['PRODUCT_TAG_NAME','T1','T2']]
tag.columns=['PRODUCT_TAG_NAME','T1','T2']
    

tag_stand=pd.read_excel('../data/tagpack/8.8机器打标目标标签.xlsx',encoding='gbk')
tag_stand.columns=['PRODUCT_TAG_NAME','T1','T2']
    

tag_stand2=pd.read_excel('../data/tagpack2/8.19 打标标签.xlsx',sheetname=0)
tag_stand2.columns=['PRODUCT_TAG_NAME','T1','T2']
    

tag_stand3=pd.read_excel('../data/tagpack0830/8.30批次标签.xlsx',sheetname=0)
tag_stand3.columns=['PRODUCT_TAG_NAME','T1','T2']
reg0=pd.read_excel('../data/tagpack0830/8.30批次标签.xlsx',sheetname=1)
reg0.columns=['PRODUCT_TAG_NAME','PRODUCT_TAG_NAME2','tag1','tag2','tag3','except']
reg0=reg0.fillna('')


tag_stand3['PRODUCT_TAG_NAME']=tag_stand3['PRODUCT_TAG_NAME'].map(reg0.set_index('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME2'].to_dict())
tag_stand3=tag_stand3.drop_duplicates()


cols=['T1','T2','PRODUCT_TAG_NAME']


tag_pd=pd.concat([tag[cols],tag_stand[cols],tag_stand2[cols],tag_stand3[cols]],axis=0)



data=pd.merge(data,tag_pd,on=cols,how='inner')



set(tag)


#tag_stand=pd.read_excel('../data/tagpack/8.8机器打标目标标签.xlsx',encoding='gbk')
#tag_stand.columns=['PRODUCT_TAG_NAME','T1','T2']
#
#
#if T_LEVEL=='T1':
#    data=pd.merge(data,tag_stand[['T1']].drop_duplicates(),on=['T1'],how='inner')
#elif T_LEVEL=='T2':
#    data=pd.merge(data,tag_stand[['T1','T2']].drop_duplicates(),on=['T1','T2'],how='inner')
#elif T_LEVEL=='T0':
#    pass
    
md=data[['PRODUCTS_NAME']].drop_duplicates()


md=md.rename(columns={'PRODUCTS_NAME':'source_text'})
md['fromLang']='auto'
md['toLang']='en'
md=md[md['source_text'].notnull()]

md.to_csv('../data/meorient_data/预注册买家映射标签20190916)_need_trans.csv',index=False,encoding='utf8')













