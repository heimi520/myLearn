#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:51:24 2019

@author: heimi
"""

import requests
import pandas as pd
import json
import datetime, time
from sqlalchemy import create_engine, types     
                  
engine_aws_bi = create_engine("oracle+cx_oracle://MEORIENTB2B_BI:MEOB2Bhs7y3bnH#G7G23VB@127.0.0.1:15212/orcl", echo=False,encoding='utf-8')
                                      
## useful table handle tools
truncate_table = lambda table_name: engine_aws_bi.execute('TRUNCATE TABLE {}'.format(table_name))
drop_table = lambda table_name: engine_aws_bi.execute('DROP TABLE {}'.format(table_name))


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
            break   ###get data ok ,sucessful out
        except Exception as e:
            print('get data error///count',c,'message',e)
            time.sleep(1)
            if c>5:
                print('try more then %s times,break out'%c)
                break  ##get data wrong ,failed to out
            
    return line_list

    

def get_aws_max_date(tb_name):
    sql="select count(*) from user_tables where table_name = '%s'"%(tb_name)
    data=pd.read_sql(sql,engine_aws_bi)
    tb_count=data.iloc[0,0]
    if tb_count>0:
        print('table %s is exsit////////////////'%tb_name)
        sql='select max("opTime") from %s '%tb_name
        data=pd.read_sql(sql,engine_aws_bi)
        dt_max_aws=data.iloc[0,0]
        return dt_max_aws
    else:
        print('table %s is not exit !!!!!!!!!'%tb_name)
        return None


tb_name='A_FACTORY_REALTIME'
timestamp_max_aws=get_aws_max_date(tb_name)



#drop_table(tb_name)
################table init data
if timestamp_max_aws is None:
    print('data table init////////////////')
    dt_st='2019-08-15 11:31:27'
    dt_ed='2019-08-18 11:31:27'
    timestamp_max_aws=timestamp_str2int(dt_st)
    timestamp_st=timestamp_max_aws
    timestamp_ed=timestamp_str2int(dt_ed)

else:
    timestamp_st=timestamp_max_aws
    dt_st=timestamp_int2str(timestamp_st)
    timestamp_ed=timestamp_str2int((pd.to_datetime(dt_st)+pd.Timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'))
    

print('dt_st',timestamp_int2str(timestamp_st),'dt_ed',timestamp_int2str(timestamp_ed) ,'aws max time',timestamp_int2str(timestamp_max_aws))

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
    print('collect data///',c,'len//',len(ret_list))
    

if len(ret_list)>0:
    print('writing add data to aws///////////')
    add_pd=pd.concat(ret_list,axis=0)
      
    idx=((add_pd['emailValid1']=='校验通过' )|(add_pd['emailValidResult1'].apply(lambda x:'unknown' in str(x))) ) &  ((add_pd['phoneValid1']=='校验通过')|(add_pd['mobileValid1'] =='校验通过'))
    add_pd=add_pd[idx]
        
#    cols=['emailValid1','emailValidResult1','phoneValid1','mobileValid1']
#    aa=add_pd[cols]
#    
    
    add_pd['opTime'].max()
    
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
        'opTime',
         
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
                    'companyName':types.VARCHAR(100),
                    'firstName':types.VARCHAR(50),
                    'lastName':types.VARCHAR(50),
                    'opTime':types.FLOAT
                    }
    
    
    add_pd.to_sql(tb_name.lower(),engine_aws_bi,index=False, if_exists='append',chunksize=500,dtype=recom_df_dtype)
    print('wrting table successful//////////////added lines//',len(add_pd))
    aa=pd.read_sql('select * from %s '%tb_name,engine_aws_bi)
#    
    
else:
    print('no add data//////////////////////////////////')



