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



import cx_Oracle
from sqlalchemy import create_engine, types
import pandas as pd
import os


#
#
#
#ORACLE_PRODUCER_CONF = {
#    'server':'20.39.240.74',
#    'port': 1521,
#    'db': 'ORCL',
#    'user': 'MEORIENTB2B_PRD_TRACK',
#    'pwd': 'Meob2b263UdR41'
#}
#




ORACLE_PRODUCER_CONF = {
    'server':'127.0.0.1',
    'port': 11521,
    'db': 'orcl',
    'user': 'MEORIENTB2B_PRD_TRACK',
    'pwd': 'Meob2b263UdR41'
}


ORACLE_TENCENT_CONF = {
    'server':'115.159.224.196',
    'port': 1521,
    'db': 'orcl',
    'user': 'MEORIENTB2B_PRES',
    'pwd': 'Meob2bZXkV4MKLyME2'
}


ORACLE_AWS_BACKUP_CONF = {
    'server':'127.0.0.1',
    'port': 15212,
    'db': 'orcl',
    'user': 'MEORIENTB2B_PRES',
    'pwd': 'MEOB2Bhs7y3bnH#G7G23VB'
    }

#52.29.102.35   ## 亚马逊云备库
#MEORIENTB2B_PRES 
#MEOB2Bhs7y3bnH#G7G23VB


def oracle_conn(conf):
    tns = cx_Oracle.makedsn(conf['server'], 
                        conf['port'], 
                        conf['db'])
    conn = cx_Oracle.connect(conf['user'], 
                        conf['pwd'], tns, 
                        encoding='utf-8')
    return conn,tns



ORACLE_AWS_BI_CONF = {
#    'server': '172.31.7.119',
    'server':'127.0.0.1',
    'port': 15212,
    'db': 'orcl',
    'user': 'MEORIENTB2B_BI',
    'pwd': 'MEOB2Bhs7y3bnH#G7G23VB'
}


#conn_producer=oracle_conn(ORACLE_PRODUCER_CONF)
conn_tencent,tns_tencent=oracle_conn(ORACLE_TENCENT_CONF)
#
conn_aws_bi,tns_bi=oracle_conn(ORACLE_AWS_BI_CONF)

#sql='SELECT * FROM MEORIENTB2B_PRD.PCM_PURCHASE_INTERESTED_TAG WHERE  ROWNUM<=100'
sql='SELECT * FROM MEORIENTB2B_PRES.PCM_PURCHASE_INTERESTED_TAG WHERE  ROWNUM<=100'

sql="""
SELECT
EM.country_name,
bb.purchaser_id,
ebi.email,
EBL.ACTION_TIME,
EBL.ACTION_TYPE 
FROM
	EM_BADGE_ACTION_LOG ebl
	LEFT JOIN EM_BADGE_INFO ebi ON EBI.id = EBL.BADGE_ID
	LEFT JOIN EM_BUYER_BADGE bb ON bb.BADGE_ID = ebl.BADGE_ID
	LEFT JOIN em_exhibition EM ON ebl.EXHIBITION_ID = EM.id 
WHERE
	EM.year = 2019 
	AND EM.period IN ( 'Q1' ) 
	AND (
		( EBL.ACTION_TYPE IN ( 'CheckIn', 'CheckOut' ) AND EBI.BADGE_TYPE_ID IN ( '0007' ) ) 
		OR ( EBL.ACTION_TYPE IN ( 'CheckIn', 'Print', 'CheckOut' ) AND EBI.BADGE_TYPE_ID IN ( '0006', '0009' ) ) 
	) 
	AND  EBL.ACTION_TIME  BETWEEN TO_DATE( '2019-06-07 05:00:00', 'yyyy-MM-dd hh24:mi:ss' )  AND TO_DATE( '2019-06-07 06:00:00', 'yyyy-MM-dd hh24:mi:ss' ) 

"""

import time
t1=time.time()
line_list=[]
for v in pd.read_sql(sql,conn_tencent,chunksize=1000):
#for v in pd.read_sql(sql,conn_tencent,chunksize=1000):
    line_list.append(v)
line_pd=pd.concat(line_list)
#line_pd=pd.read_sql(sql,conn_tencent)
t2=time.time()

print('takes time',t2-t1)


ORACLE_AWS_BI_CONF = {
#    'server': '172.31.7.119',
    'server':'127.0.0.1',
    'port': 15212,
    'db': 'orcl',
    'user': 'MEORIENTB2B_BI',
    'pwd': 'MEOB2Bhs7y3bnH#G7G23VB'
}

ORACLE_TENCENT_CONF = {
    'server':'115.159.224.196',
    'port': 1521,
    'db': 'orcl',
    'user': 'MEORIENTB2B_PRES',
    'pwd': 'Meob2bZXkV4MKLyME2'
}


ORACLE_AWS_BACKUP_CONF = {
    'server':'127.0.0.1',
    'port': 15212,
    'db': 'orcl',
    'user': 'MEORIENTB2B_PRES',
    'pwd': 'MEOB2Bhs7y3bnH#G7G23VB'
    }

from sqlalchemy import   create_engine 

engine_aws_bi = create_engine("oracle+cx_oracle://MEORIENTB2B_BI:MEOB2Bhs7y3bnH#G7G23VB@127.0.0.1:15212/orcl",encoding='utf-8', echo=True)
                     
engine_aws_backup= create_engine("oracle+cx_oracle://MEORIENTB2B_PRES:MEOB2Bhs7y3bnH#G7G23VB@127.0.0.1:15212/orcl",encoding='utf-8', echo=False)

engine_tencent = create_engine("oracle+cx_oracle://MEORIENTB2B_PRES:Meob2bZXkV4MKLyME2@115.159.224.196:1521/orcl",encoding='utf-8', echo=False)


## useful table handle tools
truncate_table = lambda table_name: engine_aws_bi.execute('TRUNCATE TABLE {}'.format(table_name))
drop_table = lambda table_name: engine_aws_bi.execute('DROP TABLE {}'.format(table_name))


truncate_table('PRODUCERDB_ONLINE_UPDATE')
drop_table('PRODUCERDB_ONLINE_UPDATE')


#aa=pd.read_sql(sql,engine_tencent)

#cc=pd.read_sql(sql,engine_aws_backup)  

#
#conn=engine_aws_backup.connect()
#result=conn.execute(sql)
#
#
#conn.close()

    
    
from sqlalchemy import create_engine, types                       
#MEORIENTB2B_BI.

recom_df_dtype = {c: types.VARCHAR(100) for c in  line_pd.columns}

recom_df_dtype={'COUNTRY_NAME':types.VARCHAR(50), 'PURCHASER_ID':types.VARCHAR(50),
                'EMAIL':types.VARCHAR(100), 'ACTION_TIME':types.DATE, 'ACTION_TYPE':types.VARCHAR(10)}

line_pd.to_sql('A_TEMP_ONLINE_UPDATE11',engine_aws_bi,index=False, if_exists='append',chunksize=1000,dtype=recom_df_dtype)


aa=pd.read_sql('select * from A_TEMP_ONLINE_UPDATE9 where rownum<=100',engine_aws_bi)



drop_table('A_TEMP_ONLINE_UPDATE11')






















































