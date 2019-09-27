#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 13:37:30 2019

@author: heimi
"""


from config import *
import warnings
warnings.filterwarnings('ignore')
import time
from sqlalchemy import create_engine, types
import pandas as pd
import requests
import datetime
import logging
import re    
import os

from email_log import *



root='logs_monitor'
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

tb_name=factory_tb_name
tb_match_name=match_tb_name


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





def get_factory_data_page(page,size,timestamp_st,timestamp_ed):
    c=0
    while True:
        c+=1
        try:
            if RUN_MODE=='test':
                url='http://10.21.64.21:9098/service/factorydata/queryCheckinByOptimePage?page=%s&size=%s&timeStart=%s&timeEnd=%s'%(page,size,timestamp_st,timestamp_ed)
            else:
                url='http://40.89.189.173:9098/service/factorydata/queryCheckinByOptimePage?page=%s&size=%s&timeStart=%s&timeEnd=%s'%(page,size,timestamp_st,timestamp_ed)
            
            logging.info(url)
            response=requests.get(url)
            data_dict=response.json()
            line_list=data_dict['payload']['ret']
            return line_list

        except Exception as e:
            logging.info('get request data error///count:%s  message:%s'%(c,e))
            time.sleep(1)
            if c>5:
                logging.info('request try more then %s times,bad  break out'%c)
                return None


def get_added_data(timestamp_st,timestamp_ed):
    ret_list=[]
    c=0
    while True:
        c+=1
        size=1000
        logging.info('loop page:%s'%c)
        line_list=get_factory_data_page(c,size,timestamp_st,timestamp_ed)
        
        if line_list is None:
            logging.info('loop request data is none')
            break
            
        logging.info('loop request data len////%s'%len(line_list))   
        line_pd=pd.DataFrame(line_list)

        if len(line_pd)==0:
            break
        line_pd=line_pd.rename(columns={'opTime':'op_time'})
        line_pd['op_time']=line_pd['op_time'].astype(int)
        line_pd['optime_str']=line_pd['op_time'].apply(timestamp_int2str)
        
        logging.info('loop batch data len:%s'%len(line_pd))
        logging.info('loop op min:%s ,timestamp_st:%s '%( timestamp_int2str(line_pd['op_time'].min()),
                                                    timestamp_int2str(timestamp_st)  ))
        
        
        line_pd=line_pd[line_pd['op_time']>timestamp_st]
        
        logging.info('loop time filter len:%s'%len(line_pd))
        if len(line_pd)==0:
            break
    
        ret_list.append(line_pd)
        logging.info('loop collect  request data///:%s'%(c))
        time.sleep(0.5)
    return ret_list


def timestamp_str2int(datetime_str):
    datetime_obj = datetime.datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
     # 10位，时间点相当于从1.1开始的当年时间编号
    date_stamp = str(int(time.mktime(datetime_obj.timetuple())))
    # 3位，微秒
    data_microsecond = str("%06d" % datetime_obj.microsecond)[0:3]
    date_stamp = date_stamp + data_microsecond
    return int(date_stamp)


# 输入毫秒级的时间，转出正常格式的时间
def timestamp_int2str(timeNum):
    timeStamp = float(timeNum/1000)
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)
    return otherStyleTime


def extract_tag(col_list):
    tag_list=[]
    for line in col_list:
        line=re.sub('[\s]*[\s]',' ',line)  ##drop multi space
        line=re.sub('(^\s*)|(\s*$)','',line) ##delete head tail space
        tag_dict={}
        for vv in line.split('],'):
            vv=vv.replace('[','').replace(']','')
            sub_list=vv.split(':')
            tag_dict[sub_list[0]]=sub_list[1].split(',')
        tag_list.append(tag_dict)
    return tag_list


def extract_tag_websiteid(tag_list,add_pd):
    ret_list=[]
    for tag_dict,row in zip(tag_list,add_pd.values.tolist()):
        for id_,tag_line in tag_dict.items():
            for tag in tag_line:
                row_new=row.copy()
                row_new.extend([id_,tag])
                ret_list.append(row_new)

    ret_pd=pd.DataFrame(ret_list,columns=add_pd.columns.tolist()+['website_id','tag_code'] )
    return ret_pd


conf_pred=CONF_PRED()
conf_bi=CONF_BI()


DEVELOPERS = ['panzhigen@meorient.com','jiangtingting@meorient.com','mijiaqi@meorient.com']


engine_aws_bi = create_engine("oracle+cx_oracle://{0}:{1}@{2}:{3}/{4}".format(conf_bi.user,conf_bi.passwd,conf_bi.ip,
                                    conf_bi.port,conf_bi.db ),encoding='utf-8', echo=ECHO) 

#
#dt_ed=pd.to_datetime('2019-09-16 10:15:02')

#dt_st=pd.to_datetime('2019-09-17 09:10:00')

dt_ed=pd.datetime.now()
dt_st=dt_ed-pd.Timedelta( minutes=5)

#dt_ed=dt_ed-pd.Timedelta( minutes=2)
#dt_st=dt_ed-pd.Timedelta( minutes=5)

dt_st=dt_st.strftime('%Y-%m-%d %H:%M:%S')
dt_ed=dt_ed.strftime('%Y-%m-%d %H:%M:%S')


timestamp_st= timestamp_str2int(dt_st)
timestamp_ed= timestamp_str2int(dt_ed)

logging.info('dt_st:%s  dt_ed:%s'%(dt_st,dt_ed))

ret_list=get_added_data(timestamp_st,timestamp_ed)




if len(ret_list)>0:
    if len(ret_list)>1:
        add_pd=pd.concat(ret_list,axis=0)
    else:
        add_pd=ret_list[0]
    
    add_pd['action_time']=pd.to_datetime(add_pd['op_time'].apply(lambda x:timestamp_int2str(x)))
    
    add_pd=add_pd[add_pd['buyerProdTag'].apply(lambda x:'[' in str(x))]
    
    tag_list=extract_tag(add_pd['buyerProdTag'])
    ret_pd=extract_tag_websiteid(tag_list,add_pd)
    logging.info('url data count///%s'%len(ret_pd))
    print('action time max////',ret_pd['action_time'].max(),'ret_pd shape///',ret_pd.shape)


sql_add="""
SELECT  count(*) as cnt FROM  MEORIENTB2B_BI.%s WHERE ACTION_TIME> TO_DATE( '%s'  , 'yyyy-MM-dd hh24:mi:ss')
and ACTION_TIME < TO_DATE( '%s' , 'yyyy-MM-dd hh24:mi:ss')
"""%(tb_name,dt_st,dt_ed)

cnt_add=pd.read_sql(sql_add,engine_aws_bi)
logging.info('cnt_add///%s'%cnt_add.iloc[0,0])

sql_match="""
SELECT  count(*) as cnt FROM  MEORIENTB2B_BI.%s WHERE ACTION_TIME> TO_DATE( '%s'  , 'yyyy-MM-dd hh24:mi:ss')
and ACTION_TIME < TO_DATE( '%s' , 'yyyy-MM-dd hh24:mi:ss')
"""%(tb_match_name,dt_st,dt_ed)

cnt_match=pd.read_sql(sql_match,engine_aws_bi)


logging.info('cnt_match///%s'%cnt_match.iloc[0,0])



























































