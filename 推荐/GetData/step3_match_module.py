#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 10:33:24 2019

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


ECHO=True

engine_producer = create_engine("oracle+cx_oracle://MEORIENTB2B_PRD_TRACK:Meob2b263UdR41@127.0.0.1:11521/orcl",encoding='utf-8', echo=ECHO) 
engine_aws_bi = create_engine("oracle+cx_oracle://MEORIENTB2B_BI:MEOB2Bhs7y3bnH#G7G23VB@127.0.0.1:15212/orcl",encoding='utf-8', echo=ECHO)                     
engine_aws_backup= create_engine("oracle+cx_oracle://MEORIENTB2B_PRES:MEOB2Bhs7y3bnH#G7G23VB@127.0.0.1:15212/orcl",encoding='utf-8', echo=ECHO)
engine_tencent = create_engine("oracle+cx_oracle://MEORIENTB2B_PRES:Meob2bZXkV4MKLyME2@115.159.224.196:1521/orcl",encoding='utf-8', echo=ECHO)


tb_name='A_PRODUCER_REALTIME'


#
#for v in pd.read_sql('select * from %s '%tb_name,engine_aws_bi ,chunksize=100):
#    td=v
#    break


sql="""
SELECT   a.buyer_website_id,a.PURCHASER_ID, a.supplier_id,MAX(match_score ) as match_score   FROM
(

SELECT   BS.buyer_website_id,PR.PURCHASER_ID, BS.supplier_id,   BS.MATCH_SCORE,' match_already' as match_source  FROM  MEORIENTB2B_BI.A_PRODUCER_REALTIME  PR
LEFT JOIN  MEORIENTB2B_BI.RECOM_BUYER_FOR_SUPPLIER BS  ON  PR.PURCHASER_ID=BS.buyer_id AND PR.WEBSITE_ID =BS.buyer_website_id
WHERE   PR.ACTION_TIME>=TO_DATE('2019-06-07 01:00:00',  'yyyy-MM-dd hh24:mi:ss' ) AND PR.ACTION_TIME<TO_DATE('2019-06-07 01:05:00',  'yyyy-MM-dd hh24:mi:ss' )

UNION ALL

SELECT   PR.WEBSITE_ID as buyer_website_id , PR.PURCHASER_ID,  TSPU.SUPPLIER_ID, TSPU.TAG_SCORE*0.6 as match_score,'tag_mach' as match_source FROM   MEORIENTB2B_BI.A_PRODUCER_REALTIME  PR  LEFT JOIN MEORIENTB2B_BI.RECOM_TAG_SPU  TSPU
ON PR.WEBSITE_ID=TSPU.WEBSITE_ID AND PR.TAG_CODE=TSPU.TAG_CODE
WHERE   PR.ACTION_TIME>=TO_DATE('2019-06-07 01:00:00',  'yyyy-MM-dd hh24:mi:ss' ) AND PR.ACTION_TIME<TO_DATE('2019-06-07 01:05:00',  'yyyy-MM-dd hh24:mi:ss' )

)a GROUP BY  a.buyer_website_id,a.PURCHASER_ID, a.supplier_id
"""



for v in pd.read_sql(sql,engine_aws_bi ,chunksize=100):
    td2=v
    break




#
#for v in pd.read_sql('select * from %s '%'RECOM_BUYER_FOR_SUPPLIER',engine_aws_bi ,chunksize=100):
#    td=v
#    break



#
#sql="""
#SELECT   PR.PURCHASER_ID, PR.SUPPLIER_ID,  BS.WEBSITE_ID, BS.MATCH_SCORE FROM A_PRODUCER_REALTIME  PR
#LEFT JOIN RECOM_BUYER_FOR_SUPPLIER BS  ON  PR.PURCHASER_ID=BS.PURCHASER_ID AND PR.WEBSITE_ID =BS.SUPPLIER_WEBSITE_ID
#WHERE   PR.ACTION_TIME>=TO_DATE('2019-06-07 01:00:00',  'yyyy-MM-dd hh24:mi:ss' ) AND PR.ACTION_TIME<TO_DATE('2019-06-07 01:05:00',  'yyyy-MM-dd hh24:mi:ss' )
#
#
#"""
#
#
#for v in pd.read_sql(sql,engine_aws_bi ,chunksize=100):
#    td2=v
#    break




























































