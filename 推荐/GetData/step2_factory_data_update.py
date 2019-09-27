#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:51:24 2019

@author: heimi
"""

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


logging.basicConfig(level = logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

ECHO=False
RUN_MODE='test'  ####'test','direct','jump'


tb_name='A_FACTORY_REALTIME'
tb_match_name='ADDED_MATCH_DATA'

t0=time.time()   
## useful table handle tools

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




def get_factory_data_page(page,timestamp_st,timestamp_ed):
    c=0
    while True:
        c+=1
        try:
#            url='http://40.89.189.173:9098/service/factorydata/queryPurchaseSourceByOptimePage?page=%s&size=%s&timeStart=%s&timeEnd=%s'%(page,size,timestamp_st,timestamp_ed)
            url='http://10.21.64.21:9098/service/factorydata/queryCheckinByOptimePage?page=%s&size=%s&timeStart=%s&timeEnd=%s'%(page,size,timestamp_st,timestamp_ed)
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
                                  
    sql="select count(*) from all_tables where table_name = '%s'"%(tb_name)
    data=pd.read_sql(sql,engine_aws_bi)
    tb_count=data.iloc[0,0]
    if tb_count>0:
        data=pd.read_sql(sql_date_max,engine_aws_bi)
        dt=data.iloc[0,0]
        return dt
    else:
        return None


def extract_tag(col_list):
    tag_list=[]
    for line in col_list:
        line='1007:[HO00007,HO00082,HO00141,HO00144],1008:[HO00007,HO00082,HO00141,HO00144]'
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
        

sql_date_max='select max(OP_TIME) from MEORIENTB2B_BI.%s '%tb_name
timestamp_max_factory=get_max_date(tb_name,sql_date_max)

################table init data
if timestamp_max_factory is None:
    logging.info('data table init')
    dt_st='2019-01-22 19:05:47'
    dt_ed='2019-08-25 23:05:47'
    timestamp_max_aws=timestamp_str2int(dt_st)
    timestamp_st=timestamp_max_aws
    timestamp_ed=timestamp_str2int(dt_ed)

else:
    timestamp_st=timestamp_max_factory
    dt_st=timestamp_int2str(timestamp_st)
    timestamp_ed=timestamp_str2int((pd.to_datetime(dt_st)+pd.Timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'))
    

logging.info('get factory data from  dt_st:%s to dt_ed:%s'%(timestamp_int2str(timestamp_st),timestamp_int2str(timestamp_ed)))

ret_list=[]
c=0
while True:
    c+=1
    size=1000
    line_list=get_factory_data_page(c,timestamp_st,timestamp_ed)
    line_pd=pd.DataFrame(line_list)
    print(len(line_pd))
    
#    if len(line_pd)>0:
#        break
#    
    
    
    if len(line_pd)==0:
        break
#    
#    line_pd['opTime'].max()<timestamp_ed
#    line_pd['opTime'].min()>=timestamp_st
    line_pd['opTime_str']=line_pd['opTime'].apply(timestamp_int2str)
    line_pd=line_pd[line_pd['opTime']>timestamp_st]
    if len(line_pd)==0:
        break
    
    ret_list.append(line_pd)
    logging.info('collect  request data///:%s'%(c))
    
    


if len(ret_list)>0:
    logging.info('writing factory added data to aws///////////')
    engine_aws_bi = create_engine("oracle+cx_oracle://{0}:{1}@{2}:{3}/{4}".format(conf_bi.user,conf_bi.passwd,conf_bi.ip,
                                conf_bi.port,conf_bi.db ),encoding='utf-8', echo=ECHO) 

    add_pd=pd.concat(ret_list,axis=0)
    add_pd=add_pd[add_pd['buyerProdTag'].notnull()]
    
    tag_list=extract_tag(add_pd['buyerProdTag'])
    ret_pd=extract_tag_websiteid(tag_list,add_pd)

     #####filter condition///////////////////////
    idx=((add_pd['emailValid1']=='校验通过' )|(add_pd['emailValidResult1'].apply(lambda x:'unknown' in str(x))) )\
    &  ((add_pd['phoneValid1']=='校验通过')|(add_pd['mobileValid1'] =='校验通过'))\
    &(add_pd['companyName'].notnull())
    
    add_pd=add_pd[idx]
        
#    cols=['emailValid1','emailValidResult1','phoneValid1','mobileValid1']
#    aa=add_pd[cols]
    
    ret_pd['op_time']=ret_pd['opTime'].copy()
    ret_pd['action_time']=pd.to_datetime('19000101')
    ret_pd=ret_pd.rename(columns={'purchaseId':'purchaser_id'})
    cols=['email1',
        'emailValid1',
        'phone',
        'phoneValid1',
        'mobile',
        'mobileValid1',
        'whatsAppValid',
        'buyerProdTag',
        'companyName',
        'firstName',
        'lastName',
        'action_time',
        'op_time',
        'website_id',
        'tag_code',
        'purchaser_id'
        ]
    
    ret_pd=ret_pd[cols]
    recom_df_dtype={'email1':types.VARCHAR(100),
                    'emailValid1':types.VARCHAR(50),
                    'phone':types.VARCHAR(50),
                    'phoneValid1':types.VARCHAR(50),
                    'mobile':types.VARCHAR(50),
                    'mobileValid1':types.VARCHAR(50),
                    'whatsAppValid':types.VARCHAR(50),
                    'buyerProdTag':types.VARCHAR(50),
                    'companyName':types.VARCHAR(200),
                    'firstName':types.VARCHAR(50),
                    'lastName':types.VARCHAR(50),
                    'action_time':types.DATE,
                    'op_time':types.Numeric,
                    'website_id':types.VARCHAR(50),
                    'tag_code':types.VARCHAR(50),
                    'purchaser_id':types.VARCHAR(50),
                    }

    ret_pd.to_sql(tb_name.lower(),engine_aws_bi,index=False, if_exists='append',chunksize=500,dtype=recom_df_dtype)
    logging.info('wrting  factory added table successful//////////////added lines//:%s'%len(ret_pd))
#    aa=pd.read_sql('select * from %s '%tb_name,engine_aws_bi)
#    aa=pd.read_sql('select op_time from %s '%tb_name,engine_aws_bi)


else:
    logging.info('no data added/////')

###############################################################################
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
sql_date_match_max="select max(OP_TIME) from MEORIENTB2B_BI.%s where source='FACTORY' "%tb_match_name
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
    sql_match=sql_match.replace("'PRODUCER'","'FACTORY'")
    sql_match=re.sub("PR.ACTION_TIME.*[\)]",'%s',sql_match)


con_st='PR.OP_TIME>%s'%timestamp_st
con_ed='PR.OP_TIME<=%s'%timestamp_ed
sql_match=sql_match%(con_st,con_ed,con_st,con_ed,con_st,con_ed) 

tt1=time.time()
ret=engine_aws_bi.execute(sql_match)
tt2=time.time()
logging.info('writing  factory added match  successfull,takes time:%s'%(tt2-tt1))


t_ed=time.time()
logging.info('factory added data match takes time :%s seconds'%(t_ed-t0))


sql="select count(*) from %s"%tb_match_name
dd=pd.read_sql(sql,engine_aws_bi)
logging.info('all data lines:%s'%dd.iloc[0,0])

truncate_table = lambda table_name: engine_aws_bi.execute('TRUNCATE TABLE {}'.format(table_name))
drop_table = lambda table_name: engine_aws_bi.execute('DROP TABLE {}'.format(table_name))



