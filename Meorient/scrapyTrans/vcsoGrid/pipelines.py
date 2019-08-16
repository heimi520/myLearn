# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html


import time
import pandas as pd
import os


class VcsogridPipeline(object):
    def __init__(self):
        self.count=0
        self.t0=time.time()

    # pipeline默认调用
    def process_item(self, item, spider):
        self.count+=1
        batch_result = item['batch_result']
        fromLang = item['fromLang']
        toLang=item['toLang']
        output_path = item['output_path']
        batch_idx=item['batch_idx']
        count_total=item['count_total']
        is_batch=item['is_batch']
        t2=time.time()
        print('////////////////////////pipline///','count_total',count_total,'count',self.count,'batch_idx',batch_idx,'data shape',batch_result.shape[0],'take time',t2-self.t0)

        if self.count==1:
            batch_result.to_csv(output_path,index=False)
        else:
            batch_result.to_csv(output_path, mode='a', header=False,index=False)

