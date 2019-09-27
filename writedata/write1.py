#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 11:20:18 2019

@author: heimi
"""



import pandas as pd
import numpy as np
import os

root_data1='data1'
root_data2='data2'



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
RUN_MODE='test'  ####'test','direct','jump'

tb_name='A_PRODUCER_REALTIME'
tb_match_name='ADDED_MATCH_DATA'

db_source_name='MEORIENTB2B_PRD'
db_dest_name='MEORIENTB2B_BI'

if  RUN_MODE=='test':
    db_source_name='MEORIENTB2B_PRINT_RL'


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




engine_aws_bi = create_engine("oracle+cx_oracle://{0}:{1}@{2}:{3}/{4}".format(conf_bi.user,conf_bi.passwd,conf_bi.ip,
                            conf_bi.port,conf_bi.db ),encoding='utf-8', echo=ECHO) 


def read_csv_list(root):
    line_list=[]
    for v in os.listdir(root):
        if 'csv' in v:
            print(v)
            path=os.path.join(root,v)
            td=pd.read_csv(path)
            
            if len(td)>1:
                line_list.append(td)
    
    line_pd=pd.concat(line_list,axis=0)
    return line_pd



dd=pd.read_csv('data1/MEORIENTB2B_BI.PRODUCT_TAG_DEFINE_0.csv')
#dd=read_csv_list(root_data1)

dd.to_sql('PRODUCT_TAG_DEFINE2',engine_aws_bi,index=False, if_exists='append',chunksize=500)


















