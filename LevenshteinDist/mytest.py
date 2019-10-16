#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:07:34 2019

@author: heimi
"""

import pandas as pd
import numpy as np
import Levenshtein


def clean(line):
    line=line.lower().replace(' ','')
    return line
#dd=pd.read_excel('标签汇总-new.xlsx')
#dd[['TAG']].to_csv('tag.csv',index=False)

tag=pd.read_csv('tag.csv')
tag['dest_clean']=tag['TAG'].apply(clean)
tag['len']=tag['dest_clean'].apply(lambda x:len(x))


src_line='glass'

#str2='i do not love you'
#str3='ilove '


import time


t1=time.time()

tag_list=tag['dest_clean'].tolist()
dist_list=[]
src=clean(src_line)
len_src=len(src)
for dest_line in  tag_list:
    dist=Levenshtein.distance(src,dest_line[:len_src])
    dist_list.append(dist)
tag['src']=src_line
tag['dist']=dist_list    
min_len_pd=tag.groupby('dist')[['len']].min().reset_index()

tag=tag.sort_values('dist')

################
sort_list=[]
dist_list=list(set(tag['dist']))
for dist in dist_list:
    td=tag[tag['dist']==dist]
    td=td.sort_values('len')
    sort_list.append(td)

sort_pd=pd.concat(sort_list,axis=0)

##############

sort_list=[]
dist_list=list(set(tag['dist']))
for dist in dist_list:
    td=tag[tag['dist']==dist]
    td=td.sort_values('len')
    sort_list.append(td)


###################
   
    
    
    
t2=time.time()
print('takes time',t2-t1)


#print(dist)






