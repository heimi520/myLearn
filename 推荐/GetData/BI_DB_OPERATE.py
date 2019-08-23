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
import pandas as pd
import logging

logging.basicConfig(level = logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

ECHO=False
IS_DERECT=False  ####True is direct connect


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


#tb_name='A_PRODUCER_REALTIME'
#tb_match_name='A_PRODUCER_MATCH_DEMO'

tb_name='A_FACTORY_REALTIME'
tb_match_name='A_FACTORY_MATCH_DEMO'

truncate_table = lambda table_name: engine_aws_bi.execute('TRUNCATE TABLE {}'.format(table_name))
drop_table = lambda table_name: engine_aws_bi.execute('DROP TABLE {}'.format(table_name))

engine_aws_bi = create_engine("oracle+cx_oracle://{0}:{1}@{2}:{3}/{4}".format(conf_bi.user,conf_bi.passwd,conf_bi.ip,
                                conf_bi.port,conf_bi.db ),encoding='utf-8', echo=ECHO) 
       
drop_table(tb_name)


sql='SELECT * FROM MEORIENTB2B_BI.A_PRODUCER_MATCH_DEMO WHERE ROWNUM<=10'

#aa=pd.read_sql(sql,engine_aws_bi)

#drop_table('MEORIENTB2B_BI.ADDED_MATCH_DATA')

#drop_table(tb_name)
#drop_table(tb_match_name)

#truncate_table(tb_name)
#truncate_table(tb_match_name)

#
#drop_table(tb_match_name)
#
#truncate_table(tb_name)


#
#
#with open('sql_create_producer_match_table.sql','r') as f:
#    sql=f.read().replace('A_PRODUCER_MATCH_DATA3',tb_match_name)
#    
#engine_aws_bi.execute(sql)
#    
#    
#drop_table(tb_match_name)

#sql='select * from MEORIENTB2B_BI.A_PRODUCER_MATCH_DEMO'
#
#sql="select count(*)   from  user_tables  where table_name='A_PRODUCER_MATCH_DEMO';"
#
#sql="select count(1)   from  user_tables  where table_name='A_PRODUCER_MATCH_DEMO2'"
#dd=pd.read_sql(sql,engine_aws_bi)











