# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:04:47 2019

@author: Administrator
"""



import pandas as pd
from sklearn.model_selection import train_test_split

data_his=pd.read_csv('../data/input/ali_amazon_his.csv')
data_test=pd.read_csv('../data/input/ali_amazon_test.csv')
  
data=pd.concat([data_his,data_test],axis=0)
  

#data=pd.read_csv('../data/input/all_data_dp.csv')

#data=pd.read_csv('../data/input/data_filter.csv')

data=data[data['PRODUCT_NAME'].notnull()]
dd=data[data['PRODUCT_NAME'].apply(lambda x:len(x.replace(' ','').replace('\n',''))>=3)]

#data_his,data_test = train_test_split(dd, test_size=0.3)
#data_test.to_csv('../data/input/data_test_to_trans.csv',index=False)
data_test=pd.read_csv('../data/input/data_test_to_trans.csv')

#aa=data_test.groupby('PRODUCT_TAG_NAME')['PRODUCT_NAME'].count().sort_values().to_frame('count')



"""dropduplicate"""
#data=data.groupby(['PRODUCT_NAME']).head(1).reset_index() 

#dd=data[data['LANGUAGE']=='english']
#dd['lang_pred']='en'


import time
from MyGoogleTrans import *


js=MyGoogleTransTools()

tolang_list=['ar','es','pl','ru','tr','pt']


#fromlang='en'
#tolang='ar'
#
#subdata=dd[['PRODUCT_NAME','PRODUCT_TAG_NAME','PRODUCT_TAG_ID']].copy()
#subdata.index=range(len(subdata))
############################################################
#ret_trans_pd=subdata.copy()
#ret_trans_pd['trans_text']=subdata['PRODUCT_NAME'].copy()
#ret_trans_pd['source_text']=ret_trans_pd['trans_text'].copy()
#ret_trans_pd['idx']=range(len(ret_trans_pd))
#ret_trans_pd['langfrom']='en'
#ret_trans_pd['langto']='en'
#ret_trans_pd=ret_trans_pd[['source_text', 'idx', 'trans_text', 'langfrom', 'langto',
#       'PRODUCT_TAG_NAME', 'PRODUCT_TAG_ID']]
#ret_trans_pd.to_csv('../data/input_lang/en_trans_%s_data.csv'%'en',index=False)


##########################################################################
tolang='ar'
subdata=data_test[['PRODUCT_NAME','PRODUCT_TAG_NAME','PRODUCT_TAG_ID']].copy()
subdata.index=range(len(subdata))
batch_idx_list, batch_line_list,batch_str_list=to_batch_data(subdata)
batch_trans_list=[]
trans_batch(js,'en',tolang,batch_idx_list,batch_line_list,batch_str_list,batch_trans_list)
batch_trans_pd=pd.concat(batch_trans_list,axis=0)
ret_trans_pd=pd.merge(batch_trans_pd,subdata[['PRODUCT_TAG_NAME','PRODUCT_TAG_ID']],left_on=['idx'],right_index=True,how='left')
ret_trans_pd.to_csv('../data/input_lang/en_trans_%s_data.csv'%tolang,index=False)

 

# 
#
#import time
#from MyGoogleTrans import *
#from string import punctuation
#import re
#import collections
#import random
#
#js=MyGoogleTransTools()
##
#tolang_list=['ar','es','pl','ru','tr','en']
##tolang_list=['en']
#
#for lang in tolang_list:
#    line_list=[]
#    for lang in tolang_list:
#        td=pd.read_csv('../data/input_lang/en_trans_%s_data.csv'%lang)
#        td=td[td['trans_text'].notnull()]
#        col=td['trans_text'].copy()
#        
#        col=col.str.lower()
#        col=col.str.replace('[0-9]','') ###delete num
#    #        
#        add_punc = '.,;《》？！’ ” “”‘’@#￥% … &×（）——+【】{};；●，。°&～、|:：\n'
#        punc = punctuation + add_punc
#        col=col.apply(lambda x: re.sub(r"[{}]+".format(punc)," ",x))  ##delete dot
#    
#        col=col.str.replace('(^\s*)|(\s*$)','') ##delete head tail space
#        voc_list=[]
#        for v in col.str.split(' ').tolist():
#            voc_list.extend(v)
#        voc_list=[v for v in voc_list if len(v)>1]
#        voc_pd=pd.DataFrame(collections.Counter(voc_list).most_common() ,columns=['word','count'])
#        voc_pd=voc_pd[voc_pd['word'].apply(lambda x:len(x))>1]
#        
#        w1_list=[]
#        for v in range(2000):
#            line=random.sample(voc_list,1)[0]
#            w1_list.append(line)
#        
#        w2_list=[]
#        for v in range(2000):
#            line=random.sample(voc_list,2)
#            line=' '.join(line) 
#            w2_list.append(line)
#    
#        w3_list=[]
#        for v in range(2000):
#            line=random.sample(voc_list,3)
#            line=' '.join(line) 
#            w3_list.append(line)  
#            
#        w_list=[]
#        for v in [w1_list,w2_list,w3_list]:
#            w_list.extend(v)
#            
#        
#        w_pd=pd.DataFrame(w_list,columns=['trans_text'])
#        w_pd['langto']=lang
#        w_pd.to_csv('../data/input_lang/add_data_%s.csv'%lang,index=False)
#        
#
#
#
#
#
####################################
###merge all data
#
#tolang_list=['ar','es','pl','ru','tr','en']
#td_list=[]
#for lang in tolang_list:
#    cols_list=['trans_text','langto']
#    webdata=pd.read_csv('../data/input_lang/%s_data.csv'%lang).rename(columns={'text':'trans_text','lang':'langto'})
#    trans_data=pd.read_csv('../data/input_lang/en_trans_%s_data.csv'%lang)
#    add_data=pd.read_csv('../data/input_lang/add_data_%s.csv'%lang)
#    td=pd.concat([webdata[cols_list],trans_data[cols_list],add_data[cols_list]],axis=0)
#    td_list.append(td)
#    
#td_pd=pd.concat(td_list,axis=0)
#
#td_pd.to_csv('../data/input_lang/lang_data_merge.csv',index=False)
#
#
#
#
#
#




















