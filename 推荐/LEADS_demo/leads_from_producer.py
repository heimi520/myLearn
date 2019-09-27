#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:01:31 2019

@author: heimi
"""
import os
os.environ['NLS_LANG'] = 'AMERICAN_AMERICA.AL32UTF8' # 或者（CHINESE_CHINA.AL32UTF8）

from config import *
import warnings
warnings.filterwarnings('ignore')
import time
from sqlalchemy import create_engine, types
from sqlalchemy.orm import sessionmaker
import pandas as pd
import logging
import re
import datetime

import os
root='logs'
if not os.path.exists(root):
    os.makedirs(root)
ymdh=pd.datetime.now().strftime('%Y%m%d%H')
path=os.path.join(root,'%s.log'%ymdh)

logging.basicConfig(level = logging.INFO,
               filename=path,
               filemode='a',
               format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S'

               )



####################
ECHO=False
RUN_MODE='test'  ####'test','direct','jump'

tb_name=producer_tb_name.upper()
tb_match_name=match_tb_name.upper()

db_source_name='MEORIENTB2B_PRD'
db_dest_name='MEORIENTB2B_BI'

if  RUN_MODE=='test':
    db_source_name='MEORIENTB2B2_TEST'



logging.info('////////////////////////////////////////////////////////////////////////////////////////////////////////')
logging.info('start block')
logging.info('leads type:producer')
logging.info('run_mode:%s'%RUN_MODE)
logging.info('added leads to table:%s'%tb_name)
logging.info('added leads match data to:%s'%tb_match_name)



t0=time.time()
class CONF_PRED():
    if RUN_MODE=='direct':
        ##direct#####################
        ip='20.39.240.74'
        port='1521'
        user='MEORIENTB2B_PRD_TRACK'
        passwd='Meob2b263UdR41'
        db='orcl'
      
    elif RUN_MODE=='jump':
        ip='127.0.0.1'
        port='11521'
        user='MEORIENTB2B_PRD_TRACK'
        passwd='Meob2b263UdR41'
        db='orcl'
    elif RUN_MODE=='test':
        ip='10.21.64.20'
        port='1521'
        user='MEORIENTB2B_PRINT_RL'
        passwd='MEORIENTB2B_PRINT_RL'
        db='orcl'
   
class CONF_BI():
    if RUN_MODE=='direct':
        #direct#####################
        ip='172.31.7.119'
        port='1521'
        user='MEORIENTB2B_BI'
        passwd='MEOB2Bhs7y3bnH#G7G23VB'
        db='orcl'    
    elif RUN_MODE=='jump':    
        ip='127.0.0.1'
        port='15212'
        user='MEORIENTB2B_BI'
        passwd='MEOB2Bhs7y3bnH#G7G23VB'
        db='orcl'  
    elif  RUN_MODE=='test':
        ip='10.21.64.20'
        port='1521'
        user='MEORIENTB2B_BI'
        passwd='MEORIENTB2B_BI'
        db='orcl'  
        
    

    
conf_pred=CONF_PRED()
conf_bi=CONF_BI()
    
engine_producer = create_engine("oracle+cx_oracle://{0}:{1}@{2}:{3}/{4}".format(conf_pred.user,conf_pred.passwd,conf_pred.ip,
                                conf_pred.port,conf_pred.db ),encoding='utf-8', echo=ECHO) 


truncate_table = lambda table_name: engine_aws_bi.execute('TRUNCATE TABLE {}'.format(table_name))
drop_table = lambda table_name: engine_aws_bi.execute('DROP TABLE {}'.format(table_name))
       
def get_max_date(tb_name,sql_date_max):
    """
    get max date from aws data base
    """                 
    engine_aws_bi = create_engine("oracle+cx_oracle://{0}:{1}@{2}:{3}/{4}".format(conf_bi.user,conf_bi.passwd,conf_bi.ip,
                                conf_bi.port,conf_bi.db ),encoding='utf-8', echo=ECHO) 
                                  
    try:
        data=pd.read_sql(sql_date_max,engine_aws_bi)
        dt=data.iloc[0,0]
        if dt is None:
            return dt
        else:
            dt_max_aws=dt.strftime('%Y-%m-%d %H:%M:%S')
            return dt_max_aws
    except Exception as e:
        logging.info('get max date error///:%s'%e)
        if 'TNS' in str(e):
            assert(False)
        return None

    

# 输入毫秒级的时间，转出正常格式的时间
def timestamp_int2str(timeNum):
    timeStamp = float(timeNum/1000)
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime
    

def timestamp_str2int(datetime_str):
    datetime_obj = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
     # 10位，时间点相当于从1.1开始的当年时间编号
    date_stamp = str(int(time.mktime(datetime_obj.timetuple())))
    # 3位，微秒
    data_microsecond = str("%06d" % datetime_obj.microsecond)[0:3]
    date_stamp = date_stamp + data_microsecond
    return int(date_stamp)



def read_data(sql,engine_producer,chunksize=5000):
    t1=time.time()
    line_list=[]
    for k,v in enumerate(pd.read_sql(sql,engine_producer,chunksize=chunksize)):
        line_list.append(v)
        t2=time.time()
        logging.info('k///:%s  batch  data len///:%s  takes time total:%s'%(k,len(v),t2-t1))
    t2=time.time()
    return line_list

def  get_date_range(dt_max_producer):
    if dt_max_producer is None:
        logging.info('Data table init//////////')
        dt_st='2019-06-05 01:00:00'
        dt_ed='2019-06-05 02:00:00'
    else:
        dt_st=dt_max_producer
        dt_ed=(pd.to_datetime(dt_max_producer)+pd.Timedelta(days=9999)).strftime('%Y-%m-%d %H:%M:%S')
    return dt_st,dt_ed



try:
    sql_producer_date_max='select max(ACTION_TIME) from %s.%s '%(db_dest_name, producer_tb_name.upper())
    dt_max_producer=get_max_date(tb_name,sql_producer_date_max)
    
    sql_factory_date_max='select max(ACTION_TIME) from %s.%s '%(db_dest_name, factory_tb_name.upper())
    dt_max_factory=get_max_date(tb_name,sql_factory_date_max)
    
    assert(dt_max_factory is not None)  #####if dt_max_factory is None then error///////
    
    if dt_max_producer is not None :
        dt_max=max(dt_max_producer,dt_max_factory)
    else:
        dt_max=dt_max_factory
    
    logging.info('factory date max:%s ,producer date max:%s  ,date max:%s'%(dt_max_factory,dt_max_producer,dt_max))
    
    dt_st,dt_ed=get_date_range(dt_max)
    dt_max_producer=pd.to_datetime(dt_st).strftime('%Y-%m-%d %H:%M:%S')
    
    logging.info('get producer add data from dt_st:%s to dt_ed:%s'%(dt_st,dt_ed))

    with open('sql_select_producer_data.sql','r') as f:
        sql_producer_data=f.read()
    
    sql_producer_data=sql_producer_data.replace('MEORIENTB2B_PRD',db_source_name)
    sql_producer_data=re.sub('[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}','%s',sql_producer_data)
    sql_producer_data=sql_producer_data%(dt_st,dt_ed)
    
    
    tt1=time.time()
    line_list=read_data(sql_producer_data,engine_producer,chunksize=5000) 
    tt2=time.time()
    logging.info('read producer data takes time//%s'%(tt2-tt1))   
    
    
    engine_aws_bi = create_engine("oracle+cx_oracle://{0}:{1}@{2}:{3}/{4}".format(conf_bi.user,conf_bi.passwd,conf_bi.ip,
                                conf_bi.port,conf_bi.db ),encoding='utf-8', echo=ECHO) 
    
    if len(line_list)>0: 

        add_data=pd.concat(line_list,axis=0)
  
        logging.info('original lines:%s'%len(add_data))
        
        add_data['op_time']=add_data['action_time'].apply(lambda x: timestamp_str2int(x.strftime('%Y-%m-%d %H:%M:%S')))
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
    else:
        logging.info('original lines:%s'%0)
        
 
    ###########################################################
    logging.info('start wrting added match data............')
    with open('sql_create_producer_match_table.sql','r') as f:
        sql_create_match_table=f.read().replace('A_PRODUCER_MATCH_DEMO',tb_match_name)
    
    tt1=time.time()
    try:
        ret=engine_aws_bi.execute(sql_create_match_table)
    except Exception as e:
        pass
    tt2=time.time()
    logging.info('create producer match table takes time//%s'%(tt2-tt1))   
    
    #############################################################
    sql_date_match_max="select max(ACTION_TIME) from MEORIENTB2B_BI.%s  "%tb_match_name
    dt_max_match=get_max_date(tb_match_name,sql_date_match_max)      
    if dt_max_match is None:
        logging.info('match table init//////////')
        dt_match_st='1990-01-01 01:00:00'
        dt_match_ed='2199-06-05 02:00:00'
    else:
        dt_match_st=dt_max_match
        dt_match_ed=(pd.to_datetime(dt_match_st)+pd.Timedelta(days=9999)).strftime('%Y-%m-%d %H:%M:%S')
    logging.info('set match timestamp from %s to %s '%( dt_match_st,dt_match_ed))
    with open('sql_insert_producer_match_table.sql','r') as f:
        sql_match=f.read().replace('A_PRODUCER_MATCH_DEMO',tb_match_name)
        sql_match=sql_match.replace('A_PRODUCER_REALTIME',tb_name)
    sql_match=re.sub('[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}','%s',sql_match)
    sql_match=sql_match%(dt_match_st,dt_match_ed,dt_match_st,dt_match_ed,dt_match_st,dt_match_ed) 
    
    tt1=time.time()
    ret=engine_aws_bi.execute(sql_match)
    tt2=time.time()
    logging.info('writing  producer added match  successfull,takes time:%s'%(tt2-tt1))
    
    t_end=time.time()
    logging.info('producer added data match takes time :%s seconds'%(t_end-t0))
    
    sql="select count(*) from %s "%tb_match_name
    dd=pd.read_sql(sql,engine_aws_bi)
    logging.info('all data lines:%s'%dd.iloc[0,0])
    
    #
except Exception as e:
    logging.info(e)
    
logging.info('finish block')    




    