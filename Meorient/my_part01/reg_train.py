#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:57:57 2019

@author: heimi
"""

import pandas as pd

dd=pd.read_csv('../data/tagpack2/8.19 打标标签my.csv')
dd['source_text']=dd['优化后'].copy()
dd['fromLang']='auto'
dd['toLang']='zh'
dd.to_csv('../data/tagpack2/8.8机器打标标签规则trans.csv',index=False)
