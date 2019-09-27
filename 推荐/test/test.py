#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 11:43:43 2019

@author: heimi
"""

import requests


url='http://10.21.64.21:9098/service/factorydata/queryCheckinByOptimePage?page=1&size=1000&timeStart=1567841727000&timeEnd=2514611103000'

dd=requests.get(url)
print(dd.text)