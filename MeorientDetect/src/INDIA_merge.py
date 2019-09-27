#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:10:36 2019

@author: heimi
"""

import pandas as pd
import numpy as np


dd1=pd.read_excel('../data/output/inda_clean_td.xlsx')

dd2=pd.read_csv('../data/output/inda_clean_wlq.csv')


#tmp1=dd1.groupby('COMPANY_NAME')['number'].count().sort_values(ascending=False)

#
#name='KALLATRA TECHNOLOGIES PRIVATE LIMITED'
#tt=dd1[dd1['COMPANY_NAME']==name]
#
#dp1=dd1[['COMPANY_NAME']].drop_duplicates()
#dp2=dd1[['COMPANY_NAME','number']].drop_duplicates()




#
#aa=dd2.head(100)
#
#
#sub_ret.sample(1000).info()



line_list=[]
idx_list=[]
for v in dd2.groupby('SOURCE'):
    source=v[0]
    idx_list.append(source)
    
    print(source)
    td=v[1].groupby('COMPANY_CLEAN').head(1)
    td['match']=1
    
    sd=dd1.groupby('COMPANY_CLEAN').head(1)
    
    sub_ret0=pd.merge(sd,td[['COMPANY_NAME','COMPANY_CLEAN','SOURCE','match']],on=['COMPANY_CLEAN'],how='left')
    sub_ret0['match']=np.where(sub_ret0['COMPANY_CLEAN'].isnull(),0,sub_ret0['match'])
    match_num=sub_ret0['match'].sum()
    
    pct=match_num/len(sd)
    line_list.append([len(sd),len(td),match_num,pct])
    
    print('source',source,'len1',len(sd),'len2',len(td),'len merge',match_num,'pct',pct)
    
    sub_ret=pd.merge(dd1,td[['COMPANY_NAME','COMPANY_CLEAN','SOURCE','match']],on=['COMPANY_CLEAN'],how='left')
    sub_ret['match']=np.where(sub_ret['COMPANY_CLEAN'].isnull(),0,sub_ret['match'])
    
    aa=sub_ret[sub_ret['match']==1]
    bb=aa[['COMPANY_CLEAN']].drop_duplicates()
    
    
    if source=='INDEX_EXPIMP':
        break
    
#    sub_ret.to_excel('../data/output/india_clean_part_%s.xlsx'%source,index=False)
    
    

#    break

#
#line_pd=pd.DataFrame(line_list,index=idx_list,columns=['num_td','num_wlq','num_match','match_pct'])
#
#td=dd2.groupby('COMPANY_CLEAN').head(1)
#td['match']=1
#
#sd=dd1.groupby('COMPANY_CLEAN').head(1)
#
#sub_ret0=pd.merge(sd,td[['COMPANY_NAME','COMPANY_CLEAN','SOURCE','match']],on=['COMPANY_CLEAN'],how='left')
#sub_ret0['match']=np.where(sub_ret0['COMPANY_CLEAN'].isnull(),0,sub_ret0['match'])
#match_num=sub_ret0['match'].sum()
#
#pct=match_num/len(sd)



#dd1['number'].sum()





#aa=pd.merge(dd1[['COMPANY_NAME', 'COMPANY_CLEAN','number']],dd2[['COMPANY_NAME', 'COMPANY_CLEAN']],on='COMPANY_CLEAN',how='left')
#
#aa.info()


#aa.sample(10000).info()


#cols=['COMPANY_NAME', 'COMPANY_CLEAN']
#
#md1=dd1[['COMPANY_NAME', 'COMPANY_CLEAN','number']].groupby('COMPANY_CLEAN').head(1)
#md2=dd2[['COMPANY_CLEAN']].drop_duplicates()
#md2['match']=1
#
#
#md2_2=dd2[['COMPANY_NAME', 'COMPANY_CLEAN']].groupby('COMPANY_CLEAN').head(1)
#md2_2['match']=1


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


#merge2_pd=pd.merge(dd1[['COMPANY_NAME', 'COMPANY_CLEAN','number']],md2_2,on=['COMPANY_CLEAN'],how='left')
#
#merge2_pd.to_excel('../data/output/merge_no_dup_data_india_orig_show.xlsx',index=False)

#merge2_pd['number'].sum()




#merge2_pd['match'].sum()/len()



#md.to_csv('../data/output/merge_data_india.csv',index=False)


#pct=len(md)/len(md1)
























