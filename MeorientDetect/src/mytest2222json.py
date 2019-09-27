#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 18:06:25 2019

@author: heimi
"""

import pandas as pd
import json

with open('../data/input/data.txt','r',encoding='gbk') as f:
    d=f.read()
    ret=json.loads(d,encoding='gbk') 
    
   
  
dd=pd.DataFrame(ret['buckets'])
    
   