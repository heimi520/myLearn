#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:12:14 2019

@author: heimi
"""

import pandas as pd
aa=pd.read_excel('../data/output/india_export.xlsx')

aa['code']=aa['code'].apply(lambda x:x.replace('.',''))

aa.to_excel('../data/output/india_export_v2.xlsx',index=False,encoding='gbk')
