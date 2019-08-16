#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:51:05 2019

@author: heimi
"""


import os

cmd_line="""
cd /home/heimi/文档/gitWork/myLearn/test_demo/scrapyTrans
scrapy crawl vsco -a input_path=/home/heimi/文档/gitWork/myLearn/test_demo/data/pred_need_trans/small_data.csv -a output_path=/home/heimi/文档/gitWork/myLearn/test_demo/data/pred_need_trans/small_trans_data.csv

"""

os.system(cmd_line)



