# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:43:50 2019

@author: Administrator
"""

import cx_Oracle

import pandas as pd
from connect_tools import *


conn=oracle88_conn()
def read_sql(sql,conn):
    context_pd = pd.read_sql(sql,conn,chunksize=1000)
    line_list=[]
    for k,v in enumerate(context_pd):
        line_list.append(v)   
    line_pd=pd.concat(line_list,axis=0)
    return line_pd
    

########################################

sql='select * from NET.PZG_T_TAG_MAPPING'
mapdata=pd.read_sql(sql,conn)
#mapdata.to_csv('../data/input/mapdata.csv',index=False)

######################################

sql='select * from NET.PZG_PURCHASER_TAG_SYS'
buy_sys=read_sql(sql,conn)

ret_dict={}
for k,v in buy_sys.groupby('COUNTRY_NAME'):
    ret_dict[k]=v
    
   
    
#buy_sys.groupby('COUNTRY_NAME')['COUNTRY_NAME'].count().sort_values(ascending=False).head(12).index.tolist()  
    
    
country_list=['Mexico',
             'Turkey',
             'Nigeria',
             'Poland',
             'Kazakhstan',
             'United Arab Emirates',
             'Kenya',
             'South Africa',
             'Brazil',
             'Switzerland',
             'India',
             'Egypt']    
    
country_language_dict={'Mexico':'spanish',
 'Turkey':'turkish',
  'Nigeria':'english',
  'Poland':'polish',
  'Kazakhstan':'kazakh',
 'United Arab Emirates':'arabic',
 'Kenya':'english',
 'South Africa':'english',
 'Brazil':'english',
 'Switzerland':'english',
  'India':'english',
 'Egypt':'arabic'
 }    
    
    
#    break

dd_sys=buy_sys[['PRODUCT_TAG_ID','PRODUCTS_NAME','T1','T2','T3','T4']]
dd_sys_dp=dd_sys.drop_duplicates().astype(str)
#dd_sys_dp['flag']=''
#for col in dd_sys_dp.columns[:-1]:
#    dd_sys_dp['flag']+=('_'+dd_sys_dp[col])
#    

c_sys= dd_sys_dp.groupby(['PRODUCT_TAG_ID','PRODUCTS_NAME','T1','T2','T3','T4']).PRODUCT_TAG_ID.count().sort_values(ascending=False).to_frame('cnt').reset_index()
c_sys=c_sys[c_sys['cnt']==1]

cols_sys=['PRODUCTS_NAME','T1','T2','T3','T4']
ret_sys=pd.merge(c_sys[cols_sys],dd_sys_dp,on=cols_sys,how='inner')
ret_sys['SOURCE_TYPE']='sys'

if c_sys['cnt'].max()>1:
    print('bad data')
    assert(False)

sql='select * from NET.PZG_PURCHASER_TAG_MAN'
buy_man=read_sql(sql,conn)

dd_man=buy_man[['PRODUCT_TAG_ID','PRODUCTS_NAME','T1','T2','T3']]

dd_man_dp=dd_man.drop_duplicates().astype(str)
c_man= dd_man_dp.groupby(['PRODUCTS_NAME','T1','T2','T3'])['PRODUCT_TAG_ID'].count().sort_values(ascending=False).to_frame('cnt').reset_index()
c_man=c_man[c_man['cnt']==1]

cols_man=['PRODUCTS_NAME','T1','T2','T3']
ret_man=pd.merge(c_man[cols_man],dd_man_dp,on=cols_man,how='inner')
ret_man['T4']='None'
ret_man['SOURCE_TYPE']='man'

ret=pd.concat([ret_sys,ret_man],axis=0).rename(columns={'PRODUCTS_NAME':'PRODUCT_NAME'})

#ret.to_csv('../data/input/buy_data_dp.csv',index=False)

##################
sql='	SELECT * FROM net.PZG_SUPPLIER_TAG_MAN union    select * from net.PZG_SUPPLIER_TAG_SYS'
dd=read_sql(sql,conn)

#aa=dd[dd['PRODUCT_TAG_ID'].apply(lambda x:x is None)] ##PRODUCT_TAG_ID is None


dd_dp=dd[['PRODUCT_TAG_ID','PRODUCT_NAME','T1','T2','T3','T4']].drop_duplicates().astype(str)
c_sell= dd_dp.groupby(['PRODUCT_NAME','T1','T2','T3','T4'])['PRODUCT_TAG_ID'].count().sort_values(ascending=False).to_frame('cnt').reset_index()
c_sell=c_sell[c_sell['cnt']==1]

cols_sell=['PRODUCT_NAME','T1','T2','T3','T4']
ret_sell=pd.merge(c_sell[cols_sell],dd_dp,on=cols_sell,how='inner')

#ret_sell.to_csv('../data/input/data_sell_dp.csv',index=False)

#
#
#
#
#
#
#
#cols_list=['T1', 'T2', 'T3','PRODUCTS_NAME',  'PRODUCT_TAG_NAME','SOURCE_TYPE']
#dd2=dd[cols_list]
#
#dd.groupby('T1')['T1'].count()
#
#
#import nltk
#from nltk.tokenize import sent_tokenize
#text= "Good muffins cost $3.88\nin New York.  Please buy me two of them.\nThanks."
#
#aa=sent_tokenize(text)  
#
#
#
# 
#       
       













































































