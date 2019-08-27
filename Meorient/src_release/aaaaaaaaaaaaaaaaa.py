#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 11:06:21 2019

@author: heimi
"""


import pandas as pd
from nltk.corpus import nps_chat
nps_chat.fileids()


def get_other_data():
    from nltk.corpus import gutenberg
    line_list=[]
    for field in gutenberg.fileids():
        for v in gutenberg.open(field):
            line_list.append(v)
        break
    line_pd=pd.DataFrame( list(set(line_list)),columns=['PRODUCT_NAME'])
    line_pd['PRODUCT_TAG_NAME']='Other'
    line_pd['PRODUCT_TAG_ID']='Other'
    return line_pd


aa=get_other_data()



