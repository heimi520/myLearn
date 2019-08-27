#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 18:04:25 2019

@author: heimi
"""


import warnings
warnings.filterwarnings('ignore')
import sys
from os import path
sys.path.append( path.dirname(path.dirname(path.abspath(__file__))))



from scrapyTrans.main import *

execute(['scrapy', 'crawl', 'vsco','-a','data_dir=/home/heimi/文档/gitWork/myLearn/test_demo/data/pred_need_trans'])  # 你需要将此处的spider_name替换为你自己的爬虫名称



