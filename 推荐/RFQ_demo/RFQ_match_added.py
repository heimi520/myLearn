#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:51:24 2019

@author: heimi
"""
##### systemctl restart crond,service crond restart

import warnings
warnings.filterwarnings('ignore')
from os import path
import time
from sqlalchemy import create_engine, types
import pandas as pd
import requests
import datetime
import logging
import re    

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


ECHO=False
RUN_MODE='test'  ####'test','direct','jump'

tb_name='A_RFQ_REALTIME'
tb_match_name='ADDED_RFQ_MATCH_DATA'

db_source_name='MEORIENTB2B_PRD'
db_dest_name='MEORIENTB2B_BI'

if  RUN_MODE=='test':
    db_source_name='MEORIENTB2B_TEST'

logging.info('////////////////////////////////////////////////////////////////////////////////////////////////////////')
logging.info('start block')
logging.info('run_mode:%s'%RUN_MODE)
logging.info('added RFQ to table:%s'%tb_name)
logging.info('added RFQ match data to:%s'%tb_match_name)


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


def get_factory_data_page(page,size,timestamp_st,timestamp_ed):
    c=0
    while True:
        c+=1
        try:
            if RUN_MODE=='test':
                url='http://10.21.64.21:9098/service/factorydata/rfq/queryRfqByOptimePage?page=%s&size=%s&timeStart=%s&timeEnd=%s'%(page,size,timestamp_st,timestamp_ed)
            else:
                url='http://40.89.189.173:9098/service/factorydata/rfq/queryRfqByOptimePage?page=%s&size=%s&timeStart=%s&timeEnd=%s'%(page,size,timestamp_st,timestamp_ed)
            logging.info('url:%s '%url)
            response=requests.get(url)
            data_dict=response.json()
            line_list=data_dict['payload']['ret']
            return line_list

        except Exception as e:
            logging.info('get request data error///count:%s  message:%s'%(c,e))
            time.sleep(1)
            if c>5:
                logging.info('request try more then %s times,break out'%c)
                return None


def read_data(sql,engine_producer,chunksize=5000):
    logging.info('start reading data......')
    t1=time.time()
    line_list=[]
    for k,v in enumerate(pd.read_sql(sql,engine_producer,chunksize=chunksize)):
        line_list.append(v)
        t2=time.time()
        logging.info('k///:%s  batch  data len///:%s  takes time total:%s'%(k,len(v),t2-t1))
    t2=time.time()
    logging.info('reading data ok,takes time:%s'%(t2-t1))
    return line_list


def get_max_date(tb_name,sql_date_max):
    """
    get max date from aws data base
    """                 
    engine_aws_bi = create_engine("oracle+cx_oracle://{0}:{1}@{2}:{3}/{4}".format(conf_bi.user,conf_bi.passwd,conf_bi.ip,
                                conf_bi.port,conf_bi.db ),encoding='utf-8', echo=ECHO) 

    
    try:
        data=pd.read_sql(sql_date_max,engine_aws_bi)
        dt=data.iloc[0,0]
        return dt
    except Exception as e:
        logging.info('get max date error///:%s'%e)
        if 'TNS' in str(e):
            assert(False)
        return None



def get_timestamp_range(timestamp_max_factory):
    if timestamp_max_factory is None:
        logging.info('data table init')
        dt_st='2018-09-10 19:05:47'
        dt_ed='2099-12-06 23:05:47'
        timestamp_max_factory=timestamp_str2int(dt_st)
        timestamp_st=timestamp_max_factory
        timestamp_ed=timestamp_str2int(dt_ed)
    else:
        timestamp_st=timestamp_max_factory
        dt_st=timestamp_int2str(timestamp_st)
        timestamp_ed=timestamp_str2int((pd.to_datetime(dt_st)+pd.Timedelta(days=10000)).strftime('%Y-%m-%d %H:%M:%S'))
    return timestamp_st,timestamp_ed

   

def get_added_data(timestamp_st,timestamp_ed):
    ret_list=[]
    c=0
    while True:
        c+=1
        size=1000
        line_list=get_factory_data_page(c,size,timestamp_st,timestamp_ed)
        line_pd=pd.DataFrame(line_list)
        print('c////',len(line_pd))
        if len(line_pd)==0:
            break
        line_pd=line_pd.rename(columns={'opTime':'op_time'})
        line_pd['op_time']=line_pd['op_time'].astype(int)
        line_pd['optime_str']=line_pd['op_time'].apply(timestamp_int2str)
        line_pd=line_pd[line_pd['op_time']>timestamp_st]
        if len(line_pd)==0:
            break
    
        ret_list.append(line_pd)
        logging.info('collect  request data///:%s'%(c))
        time.sleep(0.5)
    return ret_list




truncate_table = lambda table_name: engine_aws_bi.execute('TRUNCATE TABLE {}'.format(table_name))
drop_table = lambda table_name: engine_aws_bi.execute('DROP TABLE {}'.format(table_name))


try:
    sql_date_max='select max(OP_TIME) from MEORIENTB2B_BI.%s '%tb_name
    timestamp_max_factory=get_max_date(tb_name,sql_date_max)
    
    timestamp_st,timestamp_ed=get_timestamp_range(timestamp_max_factory)
    logging.info('set timestamp range to   dt_st:%s to dt_ed:%s'%(timestamp_int2str(timestamp_st),timestamp_int2str(timestamp_ed)))
    
    ret_list= get_added_data(timestamp_st,timestamp_ed)
    engine_aws_bi = create_engine("oracle+cx_oracle://{0}:{1}@{2}:{3}/{4}".format(conf_bi.user,conf_bi.passwd,conf_bi.ip,
                                    conf_bi.port,conf_bi.db ),encoding='utf-8', echo=ECHO) 
    
    if len(ret_list)>0:

        ret_pd=pd.concat(ret_list,axis=0)

        
        ret_pd=ret_pd.rename(columns={'tagcode':'tag_code'})
        logging.info('original lines:%s'%len(ret_pd))
        ret_pd['RFQ_ID']=ret_pd['rfqid'].copy()
        ret_pd['RFQ_WEBSITE_ID']=ret_pd['siteid'].copy()
        ret_pd['action_time']=pd.to_datetime(ret_pd['op_time'].apply(lambda x:timestamp_int2str(x)))
        cols=[
            'op_time',
            'action_time',
            'tag_code',
            'RFQ_WEBSITE_ID',
            'RFQ_ID'
            ]
    
    
        ret_pd=ret_pd[cols]
        recom_df_dtype={
                        'op_time':types.Numeric,
                        'action_time':types.DATE,
                        'RFQ_WEBSITE_ID':types.VARCHAR(50),
                        'RFQ_ID':types.VARCHAR(50),
                        'tag_code':types.VARCHAR(50)
                        }
    
        ret_pd.to_sql(tb_name.lower(),engine_aws_bi,index=False, if_exists='append',chunksize=500,dtype=recom_df_dtype)
    else:
        logging.info('original lines:%s'%0)
        
    ###########################################################
    
    with open('RFQ_sql_create_producer_match_table.sql','r') as f:
        sql_create_match_table=f.read().replace('ADDED_RFQ_MATCH_DATA',tb_match_name)
    
    tt1=time.time()  
    try:
        ret=engine_aws_bi.execute(sql_create_match_table)
    except:
        pass
    tt2=time.time()
    tt2=time.time()
    logging.info('create RFQ match table takes time//%s'%(tt2-tt1))   
    
    #############################################################
    sql_date_match_max="select max(OP_TIME) from MEORIENTB2B_BI.%s where source='RFQ' "%tb_match_name
    dt_max_match=get_max_date(tb_match_name,sql_date_match_max) 
    logging.info('max timestamp from %s is %s'%(tb_match_name,dt_max_match))     
    if dt_max_match is None:
        logging.info('match table init//////////')
        dt_match_st=timestamp_str2int('1990-01-01 01:00:00')
        dt_match_ed=timestamp_str2int('2199-06-05 02:00:00')
    else:
        dt_match_st=dt_max_match
        dt_match_ed=timestamp_str2int((pd.to_datetime( timestamp_int2str(dt_match_st))+pd.Timedelta(days=9999)).strftime('%Y-%m-%d %H:%M:%S'))
    
    logging.info('set match timestamp from %s to %s '%(timestamp_int2str(dt_match_st),timestamp_int2str(dt_match_ed)))
    with open('RFQ_sql_insert_producer_match_table.sql','r') as f:
        sql_match=f.read().replace('ADDED_RFQ_MATCH_DATA',tb_match_name)
        sql_match=sql_match.replace('A_RFQ_REALTIME',tb_name)
        sql_match=re.sub('[0-9]{13,13}','%s',sql_match)
    
    sql_match=sql_match%(dt_match_st,dt_match_ed,dt_match_st,dt_match_ed) 
    
    tt1=time.time()
    ret=engine_aws_bi.execute(sql_match)
    tt2=time.time()
    logging.info('writing  RFQ added match  successfull,takes time:%s'%(tt2-tt1))
    
    t_ed=time.time()
    logging.info('RFQ added data match takes time :%s seconds'%(t_ed-t0))
    
    sql="select count(*) from %s"%tb_match_name
    dd=pd.read_sql(sql,engine_aws_bi)
    logging.info('match all lines:%s'%dd.iloc[0,0])

except Exception as e :
    logging.info(e)
    
    
    
logging.info('finish block')    
  
    









