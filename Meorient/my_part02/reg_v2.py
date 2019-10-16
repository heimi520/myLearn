#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 11:10:17 2019

@author: heimi
"""

import pandas  as pd



tag=pd.read_excel('../data/tagpack/8.8机器打标标签规则v2.xlsx',sheetname=1)

cols=['改后tag', 'Tag 1', 'Tag 2', 'Tag 3', 'except']

tag=tag[cols].fillna('')
tag.columns=['PRODUCT_TAG_NAME','tag1','tag2','tag3','except']







