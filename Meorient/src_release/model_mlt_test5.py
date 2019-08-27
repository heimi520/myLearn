#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 23:31:12 2019

@author: heimi
"""



import warnings
warnings.filterwarnings('ignore')
import sys
from os import path
sys.path.append( path.dirname(path.dirname(path.abspath(__file__))))


from model_config import *
from mylib.model_meorient5 import *


from DetectModel import *


import pandas as pd
from sklearn.model_selection import train_test_split



tag_args=TagModelArgs()
de_args=DetectModelArgs()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=tag_args.GPU_DEVICES # 使用编号为1，2号的GPU 
if tag_args.GPU_DEVICES=='-1':
    logger.info('using cpu')
else:
    logger.info('using gpu:%s'%tag_args.GPU_DEVICES)


import keras.backend.tensorflow_backend as KTF 
gpu_config=tf.ConfigProto()
gpu_config.gpu_options.per_process_gpu_memory_fraction=0.9 ##gpu memory up limit
gpu_config.gpu_options.allow_growth = True # 设置 GPU 按需增长,不易崩掉
session = tf.Session(config=gpu_config) 
# 设置session 
KTF.set_session(session)


print('data text clean....')




import pandas as pd
from sklearn.model_selection import train_test_split

from model_config import *
from mylib.model_meorient5 import *


tag_args=TagModelArgs()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=tag_args.GPU_DEVICES # 使用编号为1，2号的GPU 
if tag_args.GPU_DEVICES=='-1':
    logger.info('using cpu')
else:
    logger.info('using gpu:%s'%tag_args.GPU_DEVICES)


import keras.backend.tensorflow_backend as KTF 
gpu_config=tf.ConfigProto()
gpu_config.gpu_options.per_process_gpu_memory_fraction=0.9 ##gpu memory up limit
gpu_config.gpu_options.allow_growth = True # 设置 GPU 按需增长,不易崩掉
session = tf.Session(config=gpu_config) 
# 设置session 
KTF.set_session(session)





def cal_count():
    data_his=pd.read_csv('../data/input/ali_amazon_his.csv')
    data_his['flag']='train'
    data_test=pd.read_csv('../data/input/ali_amazon_test.csv')
    data_test['flag']='test'
    data_all=pd.concat([data_his,data_test],axis=0)
    data_his,data_test = train_test_split(data_all, test_size=0.05)
    
    cnt_pd=data_all.groupby('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME'].count().to_frame('count')
    return cnt_pd,data_all,data_his,data_test

cnt_pd,data,data_his,data_test=cal_count()


data_test=data


def pipeline_predict(line_list):
    col=pd.DataFrame(line_list,columns=['PRODUCT_NAME'])  
    
    text2feature=Text2Feature(seq_len=tag_args.SEQ_LEN,max_words=tag_args.MAX_WORDS,is_rename_tag=False,model_id=tag_args.MODEL_ID)

    x_test_padded_seqs=text2feature.pipeline_transform(col)
    
        
    model=TextCNN(seq_len=tag_args.SEQ_LEN,max_words=tag_args.MAX_WORDS,\
                  voc_dim=tag_args.VOC_DIM,\
                  num_classes1=text2feature.num_classes_list[0], num_classes2=text2feature.num_classes_list[1] ,
                  cut_list=text2feature.cut_list,is_pre_train=False,\
                  init_learn_rate=tag_args.INIT_LEARN_RATE,batch_size=tag_args.BATCH_SIZE,epoch_max=tag_args.EPOCH_MAX,\
                  drop_out_rate=tag_args.DROP_OUT_RATE,early_stop_count=tag_args.EARLY_STOP_COUNT,model_id=tag_args.MODEL_ID)
    
    model.build_model()
    
    [prob1,prob1_2,prob2]=model.predict(x_test_padded_seqs)

    tag1_max_int=np.argmax(prob1,axis=1)
    tag2_max_int=np.argmax(prob2,axis=1)

    [tag1_max,tag2_max]=text2feature.num2label([tag1_max_int, tag2_max_int])
    
    prob1_max=np.max(prob1,axis=1)
    prob2_max=np.max(prob2,axis=1)
    
    prob1_backup=prob1.copy()
    for k,v in enumerate(tag1_max_int):
        prob1_backup[k,v]=0
    prob2_backup=prob2.copy()
    for k,v in enumerate(tag2_max_int):
        prob2_backup[k,v]=0
    
    
    tag1_second_int=np.argmax(prob1_backup,axis=1)
    tag2_second_int=np.argmax(prob2_backup,axis=1)
    [tag1_second,tag2_second]=text2feature.num2label([tag1_second_int,tag2_second_int ])
    
    prob1_second=np.max(prob1_backup,axis=1)
    prob2_second=np.max(prob2_backup,axis=1)
    
    
        
#    tag1=pd.read_excel('../data/ali_data/服装&3C 标签7.02_new.xlsx',sheet_name=0)
#    tag2=pd.read_excel('../data/ali_data/服装&3C 标签7.02_new.xlsx',sheet_name=1)
#    tag=pd.concat([tag1,tag2],axis=0)
#    tag['PRODUCT_TAG_NAME']=tag['Product Tag'].str.replace('(^\s*)|(\s*$)','')
#    
#    tag_dict=tag.set_index('PRODUCT_TAG_NAME')['T1'].to_dict()
#    
#    tag1_max_adj=[tag_dict.get(v,'T1_Other') for v in tag1_max]
#    
#    T1_pred2=[tag_dict.get(v,'Other') for v in tag2_max]
#    
#    tmp=np.array([tag1_max,T1_pred2,tag2_max]).T
#    tag2_max_adj=np.where(tmp[:,0]!=tmp[:,1],'Other',tmp[:,-1])
#    
#    tag2_max_adj=tag2_max
    
    return tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second



#
#line_list=['mobile']
##line_list=[' samsung']
##line_list=[' iphone']
line_list=['Cellphones']

#
#line_list=['iphone']
#
#line=line_list[0]
#t=re.sub('[0-9]{1,40}[,][0-9]*','*',line)
#
#line_list=[t]

text2feature=Text2Feature(seq_len=tag_args.SEQ_LEN,max_words=tag_args.MAX_WORDS,is_rename_tag=False,model_id=tag_args.MODEL_ID)

text2feature.text_clean(line_list)
        


#aa=data_test[data_test['T1']!=data_test['T1_NAME_PRED1']]
#aa=aa[['T1','T1_NAME_PRED1','PRODUCT_TAG_NAME','TAG_NAME_PRED1','PRODUCT_NAME']]

#t=re.sub('sma[a-zA-Z]{1,5}watch','smart watch',line_list[0].lower())
#
#line_list=[t]
#print(t)


#line_list=[' White Headphones  ']  ###notice
#line_list=[' telephones ']
#line_list=['Havit i96 TWS /True Wireless Sports IPX6 Waterproof Earphone'] ####notice

#line_list=['Sport Blue Tooth Earphone']  ##notice


#line_list=['  Custom, factory, irregular, crinkled dress in the style of OEM  ']
#line_list=['phones']
#line_list=['Huawei Mate ']

#line_list=['iphone ']

line_list=['iwatch']


line_list=['huawei p20 mate10']

tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second=pipeline_predict(line_list)
print('tag2_max',tag2_max,'prob2_max',prob2_max,'tag1_max',tag1_max )



line_list=data_test['PRODUCT_NAME'].tolist()
tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second=pipeline_predict(line_list)

data_test['T1_NAME_PRED1']=tag1_max
data_test['T1_NAME_PRED2']=tag1_second
data_test['T1_prob_pred1']=prob1_max
data_test['T1_prob_pred2']=prob1_second


data_test['TAG_NAME_PRED1']=tag2_max
data_test['TAG_NAME_PRED2']=tag2_second
data_test['TAG_prob_pred1']=prob2_max
data_test['TAG_prob_pred2']=prob2_second


data_test['TAG_NAME_PRED']=data_test['TAG_NAME_PRED1']
#data_test['TAG_NAME_PRED']=np.where(data_test['TAG_prob_pred1']>=0.4,data_test['TAG_NAME_PRED1'],'Other')


sm=data_test[['PRODUCT_TAG_NAME','TAG_NAME_PRED','TAG_NAME_PRED2','PRODUCT_NAME']]



bad=sm[sm['TAG_NAME_PRED']!=sm['PRODUCT_TAG_NAME']]

bad2=bad[['PRODUCT_NAME','PRODUCT_TAG_NAME','TAG_NAME_PRED','TAG_NAME_PRED2']]

bad_list=[]
for v in bad.groupby('PRODUCT_TAG_NAME'):
    name=v[0]
    td=v[1]
    line=td.groupby('TAG_NAME_PRED')['TAG_NAME_PRED'].count().sort_values(ascending=False).to_frame('count')
    line['PRODUCT_TAG_NAME']=name
    line['TAG_NAME_PRED']=line.index
    line=line[line['count']>=5]
    line=line[['PRODUCT_TAG_NAME','TAG_NAME_PRED','count']].sort_values('count',ascending=False)
    bad_list.append(line)
#    bad_dict[name]=line

aa=data_test[data_test['TAG_NAME_PRED1']=='Other']


bad_pd=pd.concat(bad_list,axis=0)

bad_pd.to_csv('../data/output/bad_pred_stat.csv',index=False)

bad_pd=bad_pd.sort_values('count',ascending=False)


bad_data_list=[]
for (name,pred_name) in bad_pd[['PRODUCT_TAG_NAME','TAG_NAME_PRED']].values:
    line=sm[sm['PRODUCT_TAG_NAME']==name]
    line2=line[line['TAG_NAME_PRED']==pred_name]
    bad_data_list.append(line2)
#    break
    

bad_data_pd=pd.concat(bad_data_list,axis=0)
bad_data_pd=bad_data_pd[['PRODUCT_NAME','PRODUCT_TAG_NAME','TAG_NAME_PRED','TAG_NAME_PRED2']]
bad_data_pd.to_csv('../data/output/data_error_detail.csv',index=False)

#bad_pd.to_csv('../data/output/bad_count_pd_v3.csv',index=False)

data_test['cnt']=1
data_test['ok']=(data_test['TAG_NAME_PRED']==data_test['PRODUCT_TAG_NAME']).astype(int)
#############3


#aa=data_test[data_test['PRODUCT_TAG_NAME']=='Batteries']

print('acc///',data_test['ok'].sum()/len(data_test))

ret_pd=data_test.groupby('PRODUCT_TAG_NAME').agg({'ok':'sum', 'cnt': 'sum'})
ret_pd['acc']=ret_pd['ok']/ret_pd['cnt']
ret_pd['test_smaple_count']=data_test.groupby('PRODUCT_TAG_NAME')['ok'].count()
ret_pd=pd.merge(ret_pd,cnt_pd,left_index=True,right_index=True,how='left')
ret_pd=ret_pd.sort_values('acc')
ret_pd=ret_pd.rename(columns={'count':'train_sample_count'})
ret_pd['tag']=ret_pd.index

























