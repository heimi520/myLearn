#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:10:36 2019

@author: heimi
"""

import pandas as pd


dd1=pd.read_excel('../data/input/data_orig_with_number_all_v7(1).xlsx')

dd2=pd.read_csv('../data/output/印度公司名-WLQ_result.csv')


dd1['number'].sum()





#aa=pd.merge(dd1[['COMPANY_NAME', 'COMPANY_CLEAN','number']],dd2[['COMPANY_NAME', 'COMPANY_CLEAN']],on='COMPANY_CLEAN',how='left')
#
#aa.info()


#aa.sample(10000).info()




cols=['COMPANY_NAME', 'COMPANY_CLEAN']

md1=dd1[['COMPANY_NAME', 'COMPANY_CLEAN','number']].groupby('COMPANY_CLEAN').head(1)
md2=dd2[['COMPANY_CLEAN']].drop_duplicates()
md2['match']=1


md2_2=dd2[['COMPANY_NAME', 'COMPANY_CLEAN']].groupby('COMPANY_CLEAN').head(1)
md2_2['match']=1


#md=pd.merge(md1,md2,on=['COMPANY_CLEAN'],how='inner')
#
#merge_pd=pd.merge(md1,md2,on=['COMPANY_CLEAN'],how='left')
#
#merge_pd.to_excel('../data/output/merge_data_india_orig.xlsx',index=False)
#
#
#merge_pd.info()
#
#

#
#merge2_pd=pd.merge(dd1[['COMPANY_NAME', 'COMPANY_CLEAN','number']],md2,on=['COMPANY_CLEAN'],how='left')
#
#merge2_pd.to_excel('../data/output/merge_no_dup_data_india_orig.xlsx',index=False)
#
#merge2_pd['number'].sum()
#


#aa=merge2_pd.head(100)


merge2_pd=pd.merge(dd1[['COMPANY_NAME', 'COMPANY_CLEAN','number']],md2_2,on=['COMPANY_CLEAN'],how='left')

merge2_pd.to_excel('../data/output/merge_no_dup_data_india_orig_show.xlsx',index=False)

#merge2_pd['number'].sum()




#merge2_pd['match'].sum()/len()



#md.to_csv('../data/output/merge_data_india.csv',index=False)


#pct=len(md)/len(md1)
























