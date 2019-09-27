#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 16:43:28 2019

@author: heimi
"""

import requests
import pandas as pd

url_list=['http://10.21.64.21:9098/service/factorydata/queryCheckinByOptimePage?page=1&size=1000&timeStart=1568188800000&timeEnd=1568192400000',

          ]



line_list=[]
for url in url_list:
    response=requests.get(url)
    data_dict=response.json()
    try:
        line=data_dict['payload']['ret']
    except:
        break
    line_list.extend(line)
