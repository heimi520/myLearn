# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:05:29 2019

@author: Administrator
"""

#watch -n 10 nvidia-smi


import os
#cmd_line="""
#cd /home/heimi/文档/gitWork/myLearn/test_demo/scrapyTrans
#ls -l
#scrapy crawl vsco  -a data_dir=/home/heimi/文档/gitWork/myLearn/test_demo/data/pred_need_trans
#"""
#os.system(cmd_line)



cmd_line="""
cd /home/heimi/文档/gitWork/myLearn/test_demo/scrapyTrans
scrapy crawl vsco  -a input_path=/home/heimi/文档/gitWork/myLearn/test_demo/data/pred_need_trans/small_data.csv -a output_path=output_path=/home/heimi/文档/gitWork/myLearn/test_demo/data/pred_need_trans/small_trans_data2.csv
"""
os.system(cmd_line)



#os.system('scrapy crawl vsco')

#os.environ['qikang'] = 'leslie'

# os.environ['PATH']











