#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 18:22:46 2019

@author: heimi
"""


import pandas as pd
import numpy as np
import requests


dd=pd.read_csv('DC - 阿联酋公司信息处理20191010（待完成）(1).csv')


line_list=[]
for k,v in  enumerate(dd['website']):
    print('%s/%s'%(k,len(dd)))
    
    try:
        data = {
            "domain": v
            #"companyName": "G J DE MEDEIROS"
        }
        headers = {'Authorization':'Bearer AajFlue413ue4DNKx1luHa5gOo6fCMQ4'}
        req = requests.post(r'https://api.fullcontact.com/v3/company.enrich', json = data, headers = headers)
        json=req.json()
        
        
        json_pd=pd.DataFrame([json])
        json_pd['k']=k
        json_pd['url']=v
#        line_list.append([k,v,str(json)])
        
        
        line_list.append(json_pd)
        
        
        line_pd=pd.concat(line_list,axis=0)
        
        line_pd.to_csv('/home/heimi/桌面/tian_lin_jie_data_json.csv')
    except:
        ret=np.nan


#line_pd=pd.DataFrame(line_list,columns=['idx','website','result'])
#line_pd.to_csv('/home/heimi/桌面/tian_lin_jie_data_json.csv')



    

