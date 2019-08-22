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

logging.basicConfig(level = logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

ECHO=False              
   

t0=time.time()   
## useful table handle tools
#truncate_table = lambda table_name: engine_aws_bi.execute('TRUNCATE TABLE {}'.format(table_name))
#drop_table = lambda table_name: engine_aws_bi.execute('DROP TABLE {}'.format(table_name))


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
            url='http://40.89.189.173:9098/service/factorydata/queryPurchaseSourceByOptimePage?page=%s&size=%s&timeStart=%s&timeEnd=%s'%(page,size,timestamp_st,timestamp_ed)
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
    

def get_aws_max_date(tb_name):
    engine_aws_bi = create_engine("oracle+cx_oracle://MEORIENTB2B_BI:MEOB2Bhs7y3bnH#G7G23VB@127.0.0.1:15212/orcl",encoding='utf-8', echo=ECHO)      
    sql="select count(*) from user_tables where table_name = '%s'"%(tb_name)
    data=pd.read_sql(sql,engine_aws_bi)
    tb_count=data.iloc[0,0]
    if tb_count>0:
        logging.info('table %s is exsit////////////////'%tb_name)
        sql='select max("OP_TIME") from %s '%tb_name
        data=pd.read_sql(sql,engine_aws_bi)
        dt_max_aws=data.iloc[0,0]
        return dt_max_aws
    else:
        logging.info('table %s is not exit !!!!!!!!!'%tb_name)
        return None


tb_name='A_FACTORY_REALTIME'
timestamp_max_aws=get_aws_max_date(tb_name)


#drop_table(tb_name)
################table init data
if timestamp_max_aws is None:
    logging.info('data table init')
    dt_st='2019-08-15 11:31:27'
    dt_ed='2019-08-15 12:31:27'
    timestamp_max_aws=timestamp_str2int(dt_st)
    timestamp_st=timestamp_max_aws
    timestamp_ed=timestamp_str2int(dt_ed)

else:
    timestamp_st=timestamp_max_aws
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
    
    if len(line_pd)==0:
        break
    
    line_pd['opTime'].max()<timestamp_max_aws
    line_pd['opTime_str']=line_pd['opTime'].apply(timestamp_int2str)
    line_pd=line_pd[line_pd['opTime']>timestamp_st]
    if len(line_pd)==0:
        break
    
    ret_list.append(line_pd)
    logging.info('collect  request data///:%s'%(c))
    

if len(ret_list)>0:
    logging.info('writing factory added data to aws///////////')
    engine_aws_bi = create_engine("oracle+cx_oracle://MEORIENTB2B_BI:MEOB2Bhs7y3bnH#G7G23VB@127.0.0.1:15212/orcl",encoding='utf-8', echo=ECHO)      
    add_pd=pd.concat(ret_list,axis=0)
      
    idx=((add_pd['emailValid1']=='校验通过' )|(add_pd['emailValidResult1'].apply(lambda x:'unknown' in str(x))) ) &  ((add_pd['phoneValid1']=='校验通过')|(add_pd['mobileValid1'] =='校验通过'))
    add_pd=add_pd[idx]
        
#    cols=['emailValid1','emailValidResult1','phoneValid1','mobileValid1']
#    aa=add_pd[cols]

    add_pd['opTime'].max()
    add_pd['website_id']=1010
    add_pd['tag_code']=None
    add_pd['purchaser_id']='100006965782'
    add_pd['op_time']=add_pd['opTime'].copy()

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
        'op_time',##
        'website_id',
        'tag_code',
        'purchaser_id'
        ]
    
    add_pd=add_pd[cols]
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
                    'op_time':types.Numeric,
                    'website_id':types.VARCHAR(50),
                    'tag_code':types.VARCHAR(50),
                    'purchaser_id':types.VARCHAR(50),
                    }
    
    
    add_pd.to_sql(tb_name.lower(),engine_aws_bi,index=False, if_exists='append',chunksize=500,dtype=recom_df_dtype)
    logging.info('wrting  factory added table successful//////////////added lines//:%s'%len(add_pd))
#    aa=pd.read_sql('select * from %s '%tb_name,engine_aws_bi)
#    aa=pd.read_sql('select op_time from %s '%tb_name,engine_aws_bi)
    
    sql="""  
        SELECT
        	a.website_id,
        	a.PURCHASER_ID,
        	a.supplier_id,
        	MAX( match_score ) AS match_score 
        FROM
        	(
        	SELECT
        		PR.WEBSITE_ID,
        		PR.PURCHASER_ID,
        		BS.supplier_id,
        		BS.MATCH_SCORE,
        		'match_already' AS match_source 
        	FROM
        		MEORIENTB2B_BI.A_FACTORY_REALTIME  PR
        		LEFT JOIN MEORIENTB2B_BI.RECOM_BUYER_FOR_SUPPLIER BS ON PR.PURCHASER_ID = BS.buyer_id 
        		AND PR.WEBSITE_ID = BS.SUPPLIER_website_id 
        	WHERE
        		PR.op_time >=%s
        		AND PR.op_time <%s 
        		
        		UNION ALL
        		
        	SELECT
        		PR.WEBSITE_ID ,
        		PR.PURCHASER_ID,
        		TSPU.SUPPLIER_ID,
        		TSPU.TAG_SCORE * 0.6 AS match_score,
        		'tag_mach' AS match_source 
        	FROM
        		MEORIENTB2B_BI.A_FACTORY_REALTIME PR
        		LEFT JOIN MEORIENTB2B_BI.RECOM_TAG_SPU_TEMP TSPU ON PR.WEBSITE_ID = TSPU.WEBSITE_ID 
        		AND PR.TAG_CODE = TSPU.TAG_CODE  
        	WHERE
        		PR.op_time >=%s
        		AND PR.op_time < %s
        	) a 
        GROUP BY
        	a.website_id,
        	a.PURCHASER_ID,
        	a.supplier_id
        """%(timestamp_st,timestamp_ed,timestamp_st,timestamp_ed)
    sql=sql.replace('A_FACTORY_REALTIME',tb_name)
    
    ret_list=read_data(sql,engine_aws_bi,chunksize=1000)
    if len(ret_list)>0:
        logging.info('wrting factory added match data............')
        ret_pd=pd.concat(ret_list,axis=0)
        ret_pd['create_time']=pd.datetime.now()
        tb_match_name='A_FACTORY_MATCH_DATA'
        
        match_df_dtype={'website_id':types.VARCHAR(50), 
                        'purchaser_id':types.VARCHAR(50),
                        'supplier_id':types.VARCHAR(50), 
                        'match_score':types.FLOAT,
                        'create_time':types.Date
                        }
        ret_pd.to_sql(tb_match_name.lower(),engine_aws_bi,index=False, if_exists='append',chunksize=500,dtype=match_df_dtype)
        logging.info('wrting factory added match data successful///adaed lines:%s'%len(ret_pd))
        
#       b=pd.read_sql('select * from %s '%tb_match_name,engine_aws_bi)

else:
    logging.info('no add data//////////////////////////////////')



t_ed=time.time()
logging.info('producer added data match takes time :%s seconds'%(t_ed-t0))



