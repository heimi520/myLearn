#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:01:31 2019

@author: heimi
"""


import warnings
warnings.filterwarnings('ignore')
from os import path
import time
from sqlalchemy import create_engine, types
from sqlalchemy.orm import sessionmaker
import pandas as pd
import logging
import re

logging.basicConfig(level = logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

ECHO=False
IS_DERECT=False  ####True is direct connect

tb_name='A_PRODUCER_REALTIME'
tb_match_name='ADDED_MATCH_DATA'

t0=time.time()
class CONF_PRED():
    ip='127.0.0.1'
    port='11521'
    
    if IS_DERECT:
        ##direct#####################
        ip='20.39.240.74'
        port='1521'
        
    user='MEORIENTB2B_PRD_TRACK'
    passwd='Meob2b263UdR41'
    db='orcl'
  
    
    
class CONF_BI():
    ip='127.0.0.1'
    port='15212'
    
    if IS_DERECT:
        #direct#####################
        ip='172.31.7.119'
        port='1521'
    user='MEORIENTB2B_BI'
    passwd='MEOB2Bhs7y3bnH#G7G23VB'
    db='orcl'    
    
conf_pred=CONF_PRED()
conf_bi=CONF_BI()
    
engine_producer = create_engine("oracle+cx_oracle://{0}:{1}@{2}:{3}/{4}".format(conf_pred.user,conf_pred.passwd,conf_pred.ip,
                                conf_pred.port,conf_pred.db ),encoding='utf-8', echo=ECHO) 

#engine_aws_bi = create_engine("oracle+cx_oracle://MEORIENTB2B_BI:MEOB2Bhs7y3bnH#G7G23VB@127.0.0.1:15212/orcl",encoding='utf-8', echo=ECHO)                     
#engine_aws_backup= create_engine("oracle+cx_oracle://MEORIENTB2B_PRES:MEOB2Bhs7y3bnH#G7G23VB@127.0.0.1:15212/orcl",encoding='utf-8', echo=ECHO)
#engine_tencent = create_engine("oracle+cx_oracle://MEORIENTB2B_PRES:Meob2bZXkV4MKLyME2@115.159.224.196:1521/orcl",encoding='utf-8', echo=ECHO)


truncate_table = lambda table_name: engine_aws_bi.execute('TRUNCATE TABLE {}'.format(table_name))
drop_table = lambda table_name: engine_aws_bi.execute('DROP TABLE {}'.format(table_name))
       
def get_max_date(tb_name,sql_date_max):
    """
    get max date from aws data base
    """                 
    engine_aws_bi = create_engine("oracle+cx_oracle://{0}:{1}@{2}:{3}/{4}".format(conf_bi.user,conf_bi.passwd,conf_bi.ip,
                                conf_bi.port,conf_bi.db ),encoding='utf-8', echo=ECHO) 
                                  
    sql="select count(*) from all_tables where table_name = '%s'"%(tb_name)
    data=pd.read_sql(sql,engine_aws_bi)
    tb_count=data.iloc[0,0]
    if tb_count>0:
        data=pd.read_sql(sql_date_max,engine_aws_bi)
        dt=data.iloc[0,0]
        if dt is None:
            return dt
        else:
            dt_max_aws=dt.strftime('%Y-%m-%d %H:%M:%S')
            return dt_max_aws
    else:
        return None



def read_data(sql,engine_producer,chunksize=5000):
    t1=time.time()
    line_list=[]
    for k,v in enumerate(pd.read_sql(sql,engine_producer,chunksize=chunksize)):
        line_list.append(v)
        t2=time.time()
        logging.info('k///:%s  batch  data len///:%s  takes time total:%s'%(k,len(v),t2-t1))
    t2=time.time()
    return line_list


sql_date_max='select max(ACTION_TIME) from MEORIENTB2B_BI.%s '%tb_name
dt_max_producer=get_max_date(tb_name,sql_date_max)

if dt_max_producer is None:
    logging.info('Data table init//////////')
    dt_st='2019-06-05 01:00:00'
    dt_ed='2019-06-05 02:00:00'
else:
    dt_st=dt_max_producer
    dt_ed=(pd.to_datetime(dt_max_producer)+pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')

logging.info('get producer add data from dt_st:%s to dt_ed:%s'%(dt_st,dt_ed))



    
with open('sql_select_producer_data.sql','r') as f:
    sql_producer_data=f.read()
sql_producer_data=re.sub('[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}','%s',sql_producer_data)
sql_producer_data=sql_producer_data%(dt_st,dt_ed)


tt1=time.time()
line_list=read_data(sql_producer_data,engine_producer,chunksize=5000) 
tt2=time.time()
logging.info('read producer data takes time//%s'%(tt2-tt1))   

if len(line_list)>0:  
    engine_aws_bi = create_engine("oracle+cx_oracle://{0}:{1}@{2}:{3}/{4}".format(conf_bi.user,conf_bi.passwd,conf_bi.ip,
                                conf_bi.port,conf_bi.db ),encoding='utf-8', echo=ECHO) 

                              
    logging.info('start writing added producer data to aws///////////')
    add_data=pd.concat(line_list)
    add_data['op_time']=0
    cols=['country_name', 'purchaser_id', 'email', 'website_id', 'tag_code','action_time','op_time', 'action_type']
        
    add_pd=add_data[cols]
    recom_df_dtype={'country_name':types.VARCHAR(50), 
                    'purchaser_id':types.VARCHAR(100),
                    'email':types.VARCHAR(100), 
                    'tag_code':types.VARCHAR(50),
                    'website_id':types.VARCHAR(50),
                    'action_time':types.DATE,
                    'op_time':types.NUMERIC,
                    'action_type':types.VARCHAR(50)
                    }
    
    tt1=time.time()
    add_pd.to_sql(tb_name.lower(),engine_aws_bi,index=False, if_exists='append',chunksize=500,dtype=recom_df_dtype)
    tt2=time.time()
    logging.info('wrting  add  producer data ok ,added data lines:%s,takes time:%s'%(len(add_pd),tt2-tt1))
    
    if len(add_pd)>0:
        logging.info('start wrting added match data............')
        
        ###########################################################
        with open('sql_create_producer_match_table.sql','r') as f:
            sql_create_match_table=f.read().replace('A_PRODUCER_MATCH_DEMO',tb_match_name)
            
        tt1=time.time()    
        ret=engine_aws_bi.execute(sql_create_match_table)
        tt2=time.time()
        tt2=time.time()
        logging.info('create producer match table takes time//%s'%(tt2-tt1))   

        #############################################################
        sql_date_match_max="select max(ACTION_TIME) from MEORIENTB2B_BI.%s where source='PRODUCER' "%tb_match_name
        dt_max_match=get_max_date(tb_match_name,sql_date_match_max)      
        if dt_max_match is None:
            logging.info('match table init//////////')
            dt_match_st='1990-01-01 01:00:00'
            dt_match_ed='2199-06-05 02:00:00'
        else:
            dt_match_st=dt_max_match
            dt_match_ed=(pd.to_datetime(dt_match_st)+pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')

        with open('sql_insert_producer_match_table.sql','r') as f:
            sql_match=f.read().replace('A_PRODUCER_MATCH_DEMO',tb_match_name)
            sql_match=sql_match.replace('A_PRODUCER_REALTIME',tb_name)
        sql_match=re.sub('[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}','%s',sql_match)
        sql_match=sql_match%(dt_match_st,dt_match_ed,dt_match_st,dt_match_ed,dt_match_st,dt_match_ed) 
        
        tt1=time.time()
        ret=engine_aws_bi.execute(sql_match)
        tt2=time.time()
        logging.info('writing  producer added match  successfull,takes time:%s'%(tt2-tt1))

else:
    logging.info('producer data no added///////////////////')


t_end=time.time()

logging.info('producer added data match takes time :%s seconds'%(t_end-t0))










