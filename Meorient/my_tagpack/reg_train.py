#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:57:57 2019

@author: heimi
"""

import pandas as pd

dd=pd.read_csv('../data/tagpack/8.8机器打标标签规则.csv')
dd['source_text']=dd['Product Tag'].copy()
dd['fromLang']='auto'
dd['toLang']='zh'
dd.to_csv('../data/tagpack/8.8机器打标标签规则trans.csv',index=False)
