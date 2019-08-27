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
from mylib.model_meorient import *


import pandas as pd
from sklearn.model_selection import train_test_split




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



data=pd.read_csv('../data/input/data_filter.csv')

###############################################################################
tag_map=data.set_index('PRODUCT_TAG_NAME')['PRODUCT_TAG_ID'].to_dict()

tag_map.update({ 
        'Camcorders':'Video Cameras',
        'Fitness Trackers':'Smart Watches',
        'Pantyhose / Tights':'Stockings',
        'Men Vests & Waistcoats':'Men Vests',
        'Children Vests & Waistcoats':'Children Vests',
        'Women Vests & Waistcoats':'Women Vests',
        'Smart Remote Control':'Remote Control',
        'Women Fur Coats':'Women Coats',
        'Men Fur Coats':'Men Coats',
        'Children Underwear':'Brief/Underwear',
        'PDAs':np.nan,
        'Printers':np.nan,
        'Tripods':np.nan,
        'Cleaners':np.nan,
        'Buckles':np.nan,
        'Industrial Computer & Accessories':np.nan,
        'Hotel Uniforms':'Restaurant & Bar & Hotel & Promotion Uniforms',
        'Restaurant & Bar Uniforms':'Restaurant & Bar & Hotel & Promotion Uniforms',
        'Bank Uniforms':'Bank & Airline Uniforms',
        'Airline Uniforms':'Bank & Airline Uniforms',
        '3D Glasses':'VR & AR Glasses',
        'VR & AR':'VR & AR Glasses',
        'Fans & Cooling':'Fans & Cooling Pads',
        'Laptop Cooling Pads':'Fans & Cooling Pads',
        'Sewing Needles':'Sewing Needles & Threads',
        'Sewing Threads':'Sewing Needles & Threads',
        'Cassette Recorders & Players':'Cassette Players & Tapes',
        'Blank Records & Tapes':'Cassette Players & Tapes',
        'India & Pakistan Clothing':'Ethnic Clothing',
        'Asia & Pacific Islands Clothing':'Ethnic Clothing',
        'Africa Clothing':'Ethnic Clothing',
        'Traditional Chinese Clothing':'Ethnic Clothing',
        'Islamic Clothing':'Ethnic Clothing',
        
         'Laptop Bags & Cases	':'Laptop & PDA Bags & Cases',
         'PDA Bags & Cases':'Laptop & PDA Bags & Cases',
         'DVD Player Bags'	:'CD/DVD Player Bags & Cases',
         'CD Player Bags':'CD/DVD Player Bags & Cases',
         'VCD Player Bags':'CD/DVD Player Bags & Cases',
         'MP3 Bags & Cases':'MP3/MP4 Bags & Cases',
         'MP4 Bags & Cases':'MP3/MP4 Bags & Cases',
         'Home CD Players':'Home CD DVD & VCD Players',	
         'Home DVD Players':'Home CD DVD & VCD Players',
         'Home VCD Players':'Home CD DVD & VCD Players',
         'Portable CD Players	':'Portable CD DVD & VCD Players',
        'Portable DVD Players':'Portable CD DVD & VCD Players',
        'Wedding Jackets':	'Wedding Jackets / Wrap',
        'Wedding Wrap':'Wedding Jackets / Wrap',
         'Game Joysticks':'Joysticks & Game Controllers',
         'Game Controllers':'Joysticks & Game Controllers',
         
        })
    
        
data['PRODUCT_TAG_NAME']=data['PRODUCT_TAG_NAME'].map(tag_map)


data=data[data['PRODUCT_TAG_NAME'].notnull()]
data['PRODUCT_TAG_ID']=data['PRODUCT_TAG_NAME'].copy()



tag_pd=pd.read_csv('../data/output/tag_rename_v4.csv')
tag_dict=tag_pd.set_index('PRODUCT_TAG_NAME_ORIG')['PRODUCT_TAG_NAME'].to_dict()

tag_head=tag_pd.head(13)


data['PRODUCT_TAG_NAME']=data['PRODUCT_TAG_NAME'].map(tag_dict)
data['PRODUCT_TAG_ID']=data['PRODUCT_TAG_NAME'].copy()

data=data[['PRODUCT_TAG_NAME','PRODUCT_TAG_ID','PRODUCT_NAME','source','BUYSELL']]


a=data.groupby('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME'].count().to_frame('count')
a['PRODUCT_TAG_NAME']=a.index




def pipeline_predict(line_list):
    col=pd.DataFrame(line_list,columns=['PRODUCT_NAME'])  
    text2feature=Text2Feature(seq_len=tag_args.SEQ_LEN,max_words=tag_args.MAX_WORDS,is_rename_tag=False,model_id=tag_args.MODEL_ID)
    x_test_padded_seqs=text2feature.pipeline_transform(col)

#    data_test['label']=text2feature.tag2label(data_test['PRODUCT_TAG_ID'])
    
    model=TextCNN(max_words=tag_args.MAX_WORDS, model_id=tag_args.MODEL_ID)
    
    
    
#    tag_max_int=model.predict_classes(x_test_padded_seqs)
#    tag_max=text2feature.num2label(tag_max_int)
    
    y_prob=model.predict(x_test_padded_seqs)

    tag_max_int=np.argmax(y_prob,axis=1)
    tag_max=text2feature.num2label(tag_max_int)
    prob_max=np.max(y_prob,axis=1)
    
    y_prob2=y_prob.copy()
    for k,v in enumerate(tag_max_int):
        y_prob2[k,v]=0
    
    
    tag_second_int=np.argmax(y_prob2,axis=1)
    tag_second=text2feature.num2label(tag_second_int)
    
    prob_second=np.max(y_prob2,axis=1)
#

    return tag_max,tag_second,prob_max,prob_second


data_test=data

line_list=data_test['PRODUCT_NAME'].tolist()
tag_max,tag_second,prob_max,prob_second=pipeline_predict(line_list)
data_test['TAG_NAME_PRED']=tag_max
data_test['TAG_NAME_PRED2']=tag_second
data_test['prob_max']=prob_max
data_test['prob_second']=prob_second


#
#bad=pd.merge(data_test,tag_head,on=['PRODUCT_TAG_NAME'],how='inner')
#bad2=bad[bad['prob_max']<0.7]
#bad2=bad2[['PRODUCT_NAME','PRODUCT_TAG_NAME','TAG_NAME_PRED','prob_max']]
#bad2=pd.merge(bad2,tag_head[['PRODUCT_TAG_NAME','count']],on=['PRODUCT_TAG_NAME'],how='inner')
#bad2.to_csv('../data/output/bad_data2.csv')
#
#bad2.groupby('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME'].count()




#okdata=data_test[data_test['PRODUCT_TAG_NAME']==data_test['TAG_NAME_PRED']]
okdata2=data_test[data_test['prob_max']>=0.7]

retdata=okdata2[['PRODUCT_NAME','TAG_NAME_PRED']].rename(columns={'TAG_NAME_PRED':'PRODUCT_TAG_NAME'})
retdata['BUYSELL']='sell'
retdata['source']='pred'
retdata['sample_w']=1

retdata.to_csv('../data/input/data_step2.csv',index=False)

aa=okdata2.groupby('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME'].count().to_frame('count')
aa['PRODUCT_TAG_NAME']=aa.index


#
##line_list=['Pedometer Monitor Bracelet Heart Rate Body Fit Fitness Watch Smart Blood Pressure Band Q1']
#line_list=[' iphone' ]
#
#
#tag_max,tag_second,prob_max,prob_second=pipeline_predict(line_list)
#print(tag_max,tag_second,prob_max,prob_second)
#
##
#
#
#
#data_test['prob_max'].hist(bins=100)
#
#aa=data_test[data_test['prob_max']>=0.8]

d_dict={}
for v in okdata2.groupby('TAG_NAME_PRED'):
    d_dict[v[0]]=v[1]









