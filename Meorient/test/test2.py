#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 11:09:13 2019

@author: heimi
"""

import pandas as pd

dd=pd.read_excel('../data/meorient_data/8.30标签验收结果.xlsx',sheetname=3)
dd.columns=['T1_NAME_PRED', 'TAG_NAME_PRED', 'T2_NAME_PRED', 'tag_real', 'check']
dd=dd[dd['check'].notnull()]
dd['check']=dd['check'].astype(int)

aa=dd.groupby('TAG_NAME_PRED')['TAG_NAME_PRED'].count()


dd['check'].sum()/len(dd)