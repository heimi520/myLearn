#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:59:37 2019

@author: heimi
"""

import os
import pandas as pd



root='/home/heimi/文档/gitCodeLessData/myLearn/MeorientDetect/data/input/港口信息'


line_list=[]

ret_dict={}
for k,v in enumerate(os.listdir(root)):
#    print(v)
    path=os.path.join(root,v)
#    os.rename(path,os.path.join(root,'%s_port.xlsx'%k))
    dd_dict=pd.read_excel(path,None)
    
    ret_dict[v]=dd_dict
    for name,td in dd_dict.items():
        td=td.iloc[:,:11]
        td['source']=name
        if len(td)>0:
            td.columns=['Ch', 'LOCODE', 'Name', 'NameWoDiacritics', 'SubDiv', 'Function', 'Status', 'Date', 'IATA', 'Coordinates', 'Remarks', 'source']
            line_list.append(td)
    





##    break
###
line_pd=pd.concat(line_list,axis=0)

aa=line_pd.groupby('source')['source'].count().sort_values(ascending=False)

line_pd.to_excel('../data/output/data_merge.xlsx',index=False,encoding='gbk')
#
#


