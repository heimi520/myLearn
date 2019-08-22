#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:56:58 2019

@author: heimi
"""


import pandas as pd
import numpy as np
import os

root_amazon='../data/amazon_data'
root_ali='../data/ali_data'

def read_csv_list(root,source):
    line_list=[]
    for v in os.listdir(root):
        if 'csv' in v:
            print(v)
            path=os.path.join(root,v)
            td=pd.read_csv(path)
            
            if len(td)>1:
                line_list.append(td.values)
    dd_am=pd.DataFrame(np.row_stack(line_list),columns=['kw', 'mainProducts', 'number', 'productName'] if source=='ali'  else ['kw', 'productName', 'page'])
    dd_am=dd_am[['kw','productName']].drop_duplicates()
    dd_am=dd_am.rename(columns={'keyword':'PRODUCT_TAG_NAME','product':'PRODUCT_NAME'})
    dd_am['BUYSELL']='sell'
    dd_am['source']=source
    return dd_am


dd_am=read_csv_list(root_amazon,'amazon')
dd_ali=read_csv_list(root_ali,'ali')

data=pd.concat([dd_am,dd_ali],axis=0)
data=data.rename(columns={'productName':'PRODUCT_NAME','kw':'PRODUCT_TAG_NAME'})
data=data[data['PRODUCT_NAME'].notnull()]
data['PRODUCT_TAG_ID']=data['PRODUCT_TAG_NAME'].copy()


data=data[data['PRODUCT_TAG_NAME'].notnull()]
data['PRODUCT_TAG_ID']=data['PRODUCT_TAG_NAME'].copy()


tag=pd.read_excel('../data/tagpack2/8.19 打标标签.xlsx',sheetname=0)
tag.columns=['PRODUCT_TAG_NAME','T1','T2']



md=pd.merge(data,tag,on=['PRODUCT_TAG_NAME'],how='left')

md['T1']=md['T1'].fillna('T1_Other')
md['T2']=md['T2'].fillna('T2_Other')



md.sample(10000).info()


cols_list=['PRODUCT_TAG_NAME','T1', 'T2','source','PRODUCT_NAME']
md[cols_list].to_csv('../data/input/tagpack_orig_data2.csv',index=False)



md_sect=pd.merge(data,tag,on=['PRODUCT_TAG_NAME'],how='inner')


#md['T2'].unique().tolist()

#a=md.groupby('PRODUCT_TAG_NAME')['T1'].count().sort_values(ascending=False)







#tag1=pd.read_excel('../data/ali_data/服装&3C 标签7.02_new.xlsx',sheet_name=0)
#tag1['flag']='tag1'
#
#tag2=pd.read_excel('../data/ali_data/服装&3C 标签7.02_new.xlsx',sheet_name=1)
#tag2['flag']='tag2'
#
#tag3=pd.read_excel('../data/hottag/7.30预警标签+T1.xlsx',sheet_name=1).rename(columns={'t1':'T1','TAG':'Product Tag'})
#tag3['flag']='tag3'
#
#cols=['T1','Product Tag','flag']
#
#tag=pd.concat([tag1[cols],tag2[cols],tag3[cols] ],axis=0)
#tag['PRODUCT_TAG_NAME']=tag['Product Tag'].str.replace('(^\s*)|(\s*$)','')
#
##'Garment Cords' in set(tag['PRODUCT_TAG_NAME'])
#
#data_filter=pd.merge(data,tag,on=['PRODUCT_TAG_NAME'],how='inner')
#
#data_filter3=data_filter[data_filter['flag']=='tag3']
#
#    
#    
#aa_count=data_filter3.groupby('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME'].count().sort_values(ascending=True).to_frame('count')
#aa_count['tag']=aa_count.index
#
#
#cols_list=['PRODUCT_TAG_NAME','T1', 'PRODUCT_NAME', 'PRODUCT_TAG_ID', 'BUYSELL',
#       'source','flag']
#
#data_filter[cols_list].to_csv('../data/input/data_filter_hottag.csv',index=False)
##
#
#
#
#
#
#
#
#
#
#
#






