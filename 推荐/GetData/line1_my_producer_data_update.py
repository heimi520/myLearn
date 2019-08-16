#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 17:01:31 2019

@author: heimi
"""


import warnings
warnings.filterwarnings('ignore')
import sys
from os import path
import time
import cx_Oracle
from sqlalchemy import create_engine, types
import pandas as pd
import os


ECHO=False

engine_producer = create_engine("oracle+cx_oracle://MEORIENTB2B_PRD_TRACK:Meob2b263UdR41@127.0.0.1:11521/orcl",encoding='utf-8', echo=ECHO) 
engine_aws_bi = create_engine("oracle+cx_oracle://MEORIENTB2B_BI:MEOB2Bhs7y3bnH#G7G23VB@127.0.0.1:15212/orcl",encoding='utf-8', echo=ECHO)                     
engine_aws_backup= create_engine("oracle+cx_oracle://MEORIENTB2B_PRES:MEOB2Bhs7y3bnH#G7G23VB@127.0.0.1:15212/orcl",encoding='utf-8', echo=ECHO)
engine_tencent = create_engine("oracle+cx_oracle://MEORIENTB2B_PRES:Meob2bZXkV4MKLyME2@115.159.224.196:1521/orcl",encoding='utf-8', echo=ECHO)


tb_name='A_PRODUCER_REALTIME4'

def get_aws_max_date(tb_name):
    sql="select count(*) from user_tables where table_name = '%s'"%(tb_name)
    data=pd.read_sql(sql,engine_aws_bi)
    tb_count=data.iloc[0,0]
    if tb_count>0:
        print('table %s is exsit////////////////'%tb_name)
        sql='select max("ACTION_TIME") from %s '%tb_name
        data=pd.read_sql(sql,engine_aws_bi)
        dt_max_aws=data.iloc[0,0].strftime('%Y-%m-%d %H:%M:%S')
        return dt_max_aws
    else:
        print('table %s is not exit !!!!!!!!!'%tb_name)
        return None

dt_max_aws=get_aws_max_date(tb_name)

if dt_max_aws is None:
    print('data table init///////////////////////////')
    dt_st='2019-06-07 01:00:00'
    dt_ed='2019-06-07 02:00:00'

else:
    dt_st=dt_max_aws
    dt_ed=(pd.to_datetime(dt_max_aws)+pd.Timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
print('dt_st/////',dt_st,'dt_ed/////',dt_ed)


sql="""

SELECT
EM.country_name,
bb.purchaser_id,
ebi.email,
PPIT.TAG_CODE,
EBL.ACTION_TIME,
EBL.ACTION_TYPE 
FROM
 MEORIENTB2B_PRD.EM_BADGE_ACTION_LOG ebl
 LEFT JOIN MEORIENTB2B_PRD.EM_BADGE_INFO ebi ON EBI.id = EBL.BADGE_ID
 LEFT JOIN MEORIENTB2B_PRD.EM_BUYER_BADGE bb ON bb.BADGE_ID = ebl.BADGE_ID
 LEFT JOIN MEORIENTB2B_PRD.em_exhibition EM ON ebl.EXHIBITION_ID = EM.id 
 LEFT JOIN MEORIENTB2B_PRD.SYS_WEBSITE_COUNTRY swc on EM.COUNTRY_ID=swc.COUNTRY_ID
 LEFT JOIN MEORIENTB2B_PRD.PCM_PURCHASE_INTERESTED_TAG PPIT ON PPIT.PURCHASER_ID=BB.PURCHASER_ID  and PPIT.WEBSITE_ID=swc.WEBSITE_ID

 
WHERE
 EM.year = 2019 
 AND EM.period IN ( 'Q1' ) 
 AND (
  ( EBL.ACTION_TYPE IN ( 'CheckIn' ) AND EBI.BADGE_TYPE_ID IN ( '0007' ) ) 
  OR ( EBL.ACTION_TYPE IN ( 'CheckIn' ) AND EBI.BADGE_TYPE_ID IN ( '0006', '0009' ) ) 
 ) 
 AND  EBL.ACTION_TIME  >TO_DATE( '%s', 'yyyy-MM-dd hh24:mi:ss' )  AND  EBL.ACTION_TIME <= TO_DATE( '%s', 'yyyy-MM-dd hh24:mi:ss' ) 
 
 
"""%(dt_st,dt_ed)
#sql=sql.replace('\n',' ').replace('\t',' ')

t1=time.time()
line_list=[]
#for k,v in enumerate(pd.read_sql(sql,engine_tencent,chunksize=1000)):
for k,v in enumerate(pd.read_sql(sql,engine_producer,chunksize=1000)):
    line_list.append(v)
    print('k///',k,'batch  data len///',len(v))
 #line_pd=pd.read_sql(sql,conn_tencent)
t2=time.time()

print('getting data takes time',t2-t1)
    
    
if len(line_list)>0:  
    print('writing data to aws///////////')
    add_data=pd.concat(line_list)
    cols=['country_name', 'purchaser_id', 'email', 'action_time', 'action_type']
        
    add_pd=add_data[cols]
    recom_df_dtype={'country_name':types.VARCHAR(50), 
                    'purchaser_id':types.VARCHAR(50),
                    'email':types.VARCHAR(100), 
                    'action_time':types.DATE,
                    'action_type':types.VARCHAR(50)
                    }
    
    add_pd.to_sql(tb_name.lower(),engine_aws_bi,index=False, if_exists='append',chunksize=500,dtype=recom_df_dtype)
    print('wrting table successful//////////////added data lines',len(add_pd))
    #aa=pd.read_sql('select * from %s '%tb_name,engine_aws_bi)

else:
    print('producer data no added///////////////////')








































