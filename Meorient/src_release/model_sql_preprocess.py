# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 17:18:01 2019

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
  'Poland':'Polish',
  'Kazakhstan':'kazakh',
 'United Arab Emirates':'arabic',
 'Kenya':'english',
 'South Africa':'english',
 'Brazil':'english',
 'Switzerland':'english',
  'India':'english',
 'Egypt':'arabic'
 }    
    
    
def data_dropduplicate(buy_sys,buysell,sys_man):
    cols_sys=['PRODUCT_TAG_NAME','PRODUCT_TAG_ID','COUNTRY_NAME','COUNTRY_ID','PRODUCT_NAME','T1','T2','T3','T4']    
    dd_sys=buy_sys[cols_sys]
    dd_sys_dp=dd_sys.drop_duplicates().astype(str)
    
    c_sys= dd_sys_dp.groupby(cols_sys).PRODUCT_TAG_ID.count().sort_values(ascending=False).to_frame('cnt').reset_index()
    c_sys=c_sys[c_sys['cnt']==1]
    
    ret_sys=pd.merge(c_sys[cols_sys],dd_sys_dp[cols_sys],on=cols_sys,how='inner')
    ret_sys['BUYSELL']=buysell
    ret_sys['SYS_MAN']=sys_man
    return ret_sys




sql='select * from NET.PZG_PURCHASER_TAG_SYS'
buy_sys=read_sql(sql,conn)

sql="select * from NET.PZG_PURCHASER_TAG_MAN"
buy_man=read_sql(sql,conn)

sql='select * from net.PZG_SUPPLIER_TAG_SYS'
sell_sys=read_sql(sql,conn)

sql='SELECT * FROM net.PZG_SUPPLIER_TAG_MAN'
sell_man=read_sql(sql,conn)


#####################################################
buy_sys_dp=data_dropduplicate(buy_sys.rename(columns={'PRODUCTS_NAME':'PRODUCT_NAME'}),'buy','sys')
buy_man_dp=data_dropduplicate(buy_man.rename(columns={'PRODUCTS_NAME':'PRODUCT_NAME'}),'buy','man')

sell_sys_dp=data_dropduplicate(sell_sys,'sell','sys')
sell_man_dp=data_dropduplicate(sell_man,'sell','man')

ret=pd.concat([buy_man_dp,buy_sys_dp,sell_man_dp,sell_sys_dp],axis=0)

ret=ret[ret['COUNTRY_NAME'].isin(country_list)]
ret['LANGUAGE']=ret['COUNTRY_NAME'].map(country_language_dict)

ret.to_csv('../data/input/all_data_dp.csv',index=False)



















































