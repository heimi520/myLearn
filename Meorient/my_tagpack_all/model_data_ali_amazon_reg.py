#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:11:44 2019

@author: heimi
"""


import pandas as pd

import re
from string import punctuation

tag=pd.read_excel('../data/tagpack_all/标签汇总-new.xlsx',sheetname=0)
tag.columns=['PRODUCT_TAG_NAME', 'NEW Product Tag ID', '人工翻译', 'T1', 'T2', '训练日期']
tag=tag[['T1','T2','PRODUCT_TAG_NAME']]
tag=tag.drop_duplicates()


      
def noise_clean(line):
    add_punc = '.,;《》？！’ ” “”‘’@#￥% … &×（）——+【】{};；●，。°&～、|:：\n'
    punc = punctuation + add_punc
    punc=punc.replace(' ','')
    line=re.sub(r"[{}]+".format(punc)," ",line)
    line=re.sub('[\s]*[\s]',' ',line)  ##drop multi space
    line=re.sub('(^\s*)|(\s*$)','',line) ##delete head tail space
    return line


tag['TAG_CLEAN']=tag['PRODUCT_TAG_NAME'].apply(noise_clean)
  
        
ret_list=[]
idx_list=[]
for line in tag['TAG_CLEAN'].tolist():
    idx_list.append(line)
    line_list=line.split(' ')
    
    line_list=line_list+['']*(2-len(line_list))
    if len(line_list)>2:
        line_list=line_list[:1]+[','.join(line_list[1:])]
    
    line_dict={}
    for k,v in enumerate(['tag1','tag2']):
        line_dict[v]=line_list[k] 
    ret_list.append(line_dict)    
    
ret_pd=pd.DataFrame(ret_list)
ret_pd['TAG_CLEAN']=idx_list


reg=pd.merge(tag,ret_pd,on=['TAG_CLEAN'],how='left')
reg['tag3']=''
reg['except']=''

reg.to_csv('../data/tagpack_all/reg_my.csv',index=False)





















