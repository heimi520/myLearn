#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 13:54:03 2019

@author: heimi
"""


import pandas as pd
#
#en_data=pd.read_csv('../data/pred_need_trans/en_data.csv')
small_data=pd.read_csv('../data/pred_need_trans/batch_trans_data.csv')

pred_data=pd.read_csv('../data/output/tag_pred.csv')


bad01=small_data[small_data['trans_text'].isnull()]
bad02=pred_data[pred_data['TAG_NAME_PRED1']=='Other']
bad02=bad02[['PRODUCT_NAME_ORIG']].rename(columns={'PRODUCT_NAME_ORIG':'source_text'})
cols=['source_text']

bad=pd.concat([bad01[cols],bad02[cols]],axis=0)
bad['fromLang']='auto'
bad['toLang']='en'


bad.to_csv('../data/pred_need_trans/small_need_trans_single.csv',index=False)





#coldata=pd.concat([en_data,small_data],axis=0)
#coldata=coldata.rename(columns={'source_text':'PRODUCT_NAME_ORIG','trans_text':'PRODUCT_NAME'})
#




import os
cmd_line="""
cd /home/heimi/文档/gitWork/myLearn/test_demo/scrapyTrans
scrapy crawl vsco -a input_path=/home/heimi/文档/gitWork/myLearn/test_demo/data/pred_need_trans/small_data.csv -a output_path=/home/heimi/文档/gitWork/myLearn/test_demo/data/pred_need_trans/small_trans_data.csv
"""
os.system(cmd_line)



