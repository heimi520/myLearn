#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 11:10:35 2019

@author: heimi
"""


import pandas as pd

def read_orig_data():
    #data=pd.read_excel('../data/meorient_data/买家全行业映射标签（20190717）.xlsx')
    #data.to_csv('../data/meorient_data/buy_data.csv',index=False)
    data_buy=pd.read_csv('../data/meorient_data/buy_data.csv').rename(columns={'PRODUCTS_NAME':'PRODUCT_NAME'})
    data_buy['BUYSELL']='buy'
    
    data_sell=pd.read_csv('../data/meorient_data/sell_data.csv')
    data_sell['BUYSELL']='sell'
    data0=pd.concat([data_buy,data_sell],axis=0)
    
    data=data0[(data0['T1']=='Apparel')|(data0['T1']=='Consumer Electronics')]
    data=data.rename(columns={'PRODUCT_NAME':'PRODUCT_NAME_ORIG'})
    
    coldata=data[['PRODUCT_NAME_ORIG']]
    coldata=coldata.drop_duplicates()
    coldata=coldata[coldata['PRODUCT_NAME_ORIG'].notnull()]
    coldata['T1']='lang'
    return data,coldata



def read_sell_data():
    data_sell=pd.read_csv('../data/meorient_data/sell_data.csv')
    data_sell['BUYSELL']='sell'
#    data0=pd.concat([data_buy,data_sell],axis=0)

    data=data_sell.rename(columns={'PRODUCT_NAME':'PRODUCT_NAME_ORIG'})
    
    coldata=data[['PRODUCT_NAME_ORIG']]
    coldata=coldata.drop_duplicates()
    coldata=coldata[coldata['PRODUCT_NAME_ORIG'].notnull()]
    coldata['T1']='lang'
    return data,coldata





def read_trans_data():
    en_data=pd.read_csv('../data/pred_need_trans/en_data.csv')
    small_data=pd.read_csv('../data/pred_need_trans/batch_trans_data.csv')
    
    coldata=pd.concat([en_data,small_data],axis=0)
    coldata=coldata.rename(columns={'source_text':'PRODUCT_NAME_ORIG','trans_text':'PRODUCT_NAME'})

    return coldata



from  my_textrnn.rnn_config import *
from  my_textcnn.cnn_config import *
from  my_detectcnn.detect_cnn_config import *
from mylib.model_meorient_esemble import *


def Tag2T1_np(x,cut_list):
    line_list=[]
    for k, v in enumerate(cut_list):
        if k==0:
            line=x[:,:cut_list[k]]
        else:
            line=x[:,cut_list[k-1]:cut_list[k]]
        line_list.append(line)
    line_list.append(x[:,cut_list[k]:])      
    ret_list=[]
    for line in line_list:
        max_=np.reshape(np.max(line,axis=1),(-1,1))
        ret_list.append(max_)
    out=np.concatenate(ret_list, axis=1)
    return out 
    

def pipeline_predict_esemble(line_list,mode_list=['cnn','rnn']):
    """
    mode:rnn,cnn,all
    """
    rnnconfig=rnnConfig()
    cnnconfig=cnnConfig()
    col=pd.DataFrame(line_list,columns=['PRODUCT_NAME'])  
    
    text2feature=Text2Feature(rnnconfig)

    x_test_padded_seqs=text2feature.pipeline_transform(col)
    
    prob2_list=[]
    for mode in mode_list: 
        if mode=='rnn':
            rnnconfig.NUM_TAG_CLASSES=text2feature.num_classes_list[1]
            rnnconfig.TOKENIZER=text2feature.tokenizer 
            rnnconfig.CUT_LIST=text2feature.cut_list
             
            rnnmodel=TextRNN(rnnconfig)   
            rnnmodel.build_model()
            [prob1_rnn,prob2_rnn]=rnnmodel.predict(x_test_padded_seqs)
            prob2_list.append(prob2_rnn)
        elif mode=='cnn':
            cnnconfig.NUM_TAG_CLASSES=text2feature.num_classes_list[1]
            cnnconfig.TOKENIZER=text2feature.tokenizer 
            cnnconfig.CUT_LIST=text2feature.cut_list
             
            cnnmodel=TextCNN(cnnconfig)   
            cnnmodel.build_model()
            [prob1_cnn,prob2_cnn]=cnnmodel.predict(x_test_padded_seqs)
            prob2_list.append(prob2_cnn)
    
    
    for k,v in enumerate(prob2_list):
        if k==0:
            prob2_sum=v
        else:
            prob2_sum+=v
    prob2=prob2_sum/(k+1)
    prob1=Tag2T1_np(prob2,text2feature.cut_list)

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
    
    return tag1_max,tag1_second,prob1_max,prob1_second ,tag2_max,tag2_second,prob2_max,prob2_second




def detect_pipeline_predict(data_test):
    detectcnnconfig=detectcnnConfig()
    text2feature=DetectText2Feature(detectcnnconfig)
    x_test_padded_seqs=text2feature.pipeline_transform(data_test)
       
    detectcnnconfig.NUM_TAG_CLASSES=text2feature.num_classes_list[1]
    detectcnnconfig.TOKENIZER=text2feature.tokenizer 
    model=DetectTextCNN(detectcnnconfig)
       
    model.build_model() 
    prob=model.predict(x_test_padded_seqs)
    
    tag_max_int=np.argmax(prob,axis=1)
    [_,tag_max]=text2feature.num2label([tag_max_int,tag_max_int])
    
    prob_max=np.max(prob,axis=1)
    prob_backup=prob.copy()
    for k,v in enumerate(tag_max_int):
        prob_backup[k,v]=0
    
    
    tag_second_int=np.argmax(prob_backup,axis=1)
    [_,tag_second]=text2feature.num2label([tag_second_int,tag_second_int])
    
    prob_second=np.max(prob_backup,axis=1)
    
    return tag_max,tag_second,prob_max,prob_second 











