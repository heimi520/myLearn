# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:05:29 2019

@author: Administrator
"""

#watch -n 10 nvidia-smi


import warnings
warnings.filterwarnings('ignore')
import sys
from os import path
sys.path.append( path.dirname(path.dirname(path.abspath(__file__))))


from model_config import *
from mylib.model_meorient2 import *


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
gpu_config.gpu_options.per_process_gpu_memory_fraction=0.95 ##gpu memory up limit
gpu_config.gpu_options.allow_growth = True # 设置 GPU 按需增长,不易崩掉
session = tf.Session(config=gpu_config) 
# 设置session 
KTF.set_session(session)
#
import keras 
#keras.backend.set_floatx('float16')


logger.info('keras floatx:%s'%(keras.backend.floatx()))

import multiprocessing
from multiprocessing import *

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import random


def add_data_batch(k,stemer,line_list, return_dict):
    '''worker function'''
    
    ret_list=[]
    for line in line_list:
        line_list=line.split(' ')
        random.shuffle(line_list)
        line=' '.join(line_list)
        ret_list.append(line)

    return_dict[k] = ret_list


class multiProcess(Process):
    def __init__(self,data,worker):
        super().__init__()
        self.col=data
        self.worker=worker
#        
    def set_worder(self,worker):
        self.worker=worker    
        
    def to_batch_dict(self,col,num):
        batch_size=int(len(col)/num)
        batch_dict={}
        k=0
        for k in range(num-1):
            line_list=col[k*batch_size:(k+1)*batch_size]
            batch_dict[k]=line_list
        line_list=col[(k+1)*batch_size:]
        batch_dict[k+1]=line_list
        return batch_dict
            
    def run(self):
        col=self.col
        num=cpu_count()
        batch_dict=self.to_batch_dict(col,num)
        
        manager = Manager()
        return_dict = manager.dict()
        jobs = []
        
        stemer = PorterStemmer()
        for (k,line_list) in batch_dict.items():
            logger.info('start worker//%s'%k)
            p = multiprocessing.Process(target=self.worker, args=(k,stemer,line_list,return_dict,))
            jobs.append(p)
            p.start()
            
        for k,proc in enumerate(jobs):
            logger.info('join result// %s'%k)
            proc.join()
        
        ret_list=[]
        for k in range(num):
            logger.info('k///%s /// len// %s'%(k,len(return_dict[k])))
            ret_list.extend(return_dict[k])
            
        return ret_list
            




tag_pd=pd.read_csv('../data/output/tag_rename_v4.csv')
tag_dict=tag_pd.set_index('PRODUCT_TAG_NAME_ORIG')['PRODUCT_TAG_NAME'].to_dict()


data_his=pd.read_csv('../data/input/ali_amazon_his.csv')
data_test=pd.read_csv('../data/input/ali_amazon_test.csv')

data=pd.concat([data_his,data_test],axis=0)
data['PRODUCT_TAG_NAME']=data['PRODUCT_TAG_NAME'].map(tag_dict)
data['PRODUCT_TAG_ID']=data['PRODUCT_TAG_NAME'].copy()

data=data[['T1','PRODUCT_TAG_NAME','PRODUCT_TAG_ID','PRODUCT_NAME','source','BUYSELL']]
data['sample_w']=1

def get_other_data():
    from nltk.corpus import gutenberg
    import random
    line_list=[]
    for field in gutenberg.fileids()[:2]:
        for v in gutenberg.open(field):
            line_list.append(v)
    line_list=[v for v in line_list if len(re.sub('(^\s*)|(\s*$)','',v).split(' '))>5]
    line_list=random.sample(line_list,3000)
    other=pd.DataFrame( line_list,columns=['PRODUCT_NAME'])
    other['sect_flag']=1
    other['BUYSELL']='sell'
    other['source']='other'
    other['PRODUCT_TAG_ID']='Other'
    other['PRODUCT_TAG_NAME']='Other'
    other['sample_w']=1
    return other


#
#
#####################
#tokenizer=Tokenizer(num_words=None, filters='',lower=True,split=' ')  
#tokenizer.fit_on_texts(data['PRODUCT_NAME'])
#
#voc_pd=pd.DataFrame([[k,v] for (k,v) in zip(tokenizer.word_docs.keys(),tokenizer.word_docs.values())] ,columns=['key','count'])
#voc_pd=voc_pd.sort_values('count',ascending=True)
#voc_pd['pct']=voc_pd['count'].cumsum()/voc_pd['count'].sum()
#voc_pd=voc_pd[voc_pd['pct']<=0.1]
##voc_pd2=voc_pd[voc_pd['pct']>0.1]
#voc_list=voc_pd['key'].tolist()
#
##
#other_list=[]
#for v in range(10000):
#    num=random.randint(1,5)
#    sentence=' '.join(random.sample(voc_list,num))
#    other_list.append(sentence)
#    
#other=pd.DataFrame(other_list,columns=['PRODUCT_NAME'])    
#other['BUYSELL']='sell'
#other['source']='other'
#other['PRODUCT_TAG_ID']='Other'
#other['PRODUCT_TAG_NAME']='Other'
#other['sample_w']=1
#
#other.to_csv('../data/input/my_other.csv',index=False)


#def get_other_data():
#    from nltk.corpus import gutenberg
#    import random
#    line_list=[]
#    for field in gutenberg.fileids():
#        for v in gutenberg.open(field):
#            line_list.append(v)
#    line_list=[v for v in line_list if len(re.sub('(^\s*)|(\s*$)','',v).split(' '))>5]
#    line_list=random.sample(line_list,100000)
#    other=pd.DataFrame( line_list,columns=['PRODUCT_NAME'])
#    other['sect_flag']=1
#    other['BUYSELL']='sell'
#    other['source']='other'
#    other['PRODUCT_TAG_ID']='Other'
#    other['PRODUCT_TAG_NAME']='Other'
#    other['sample_w']=1
#    
#    return other
#


other=pd.read_csv('../data/input/my_other.csv')
other['T1']='T1_Other'
#other=get_other_data()
data=pd.concat([data,other],axis=0)


#other=pd.concat([other1,other2],axis=0)
#other['T1']='T1_Other'

#data=pd.concat([data,other],axis=0)

#data=data[data['PRODUCT_TAG_NAME']!='Ethnic Clothing']

cols_list=['BUYSELL', 'PRODUCT_NAME', 'PRODUCT_TAG_ID', 'T1','PRODUCT_TAG_NAME','sample_w', 'source']
data=data[cols_list]


#data['len']=data['PRODUCT_NAME'].apply(lambda x:len(x.split(' ')))
#data['len'].hist(bins=100)

import random

all_list=[]
for (name,td) in data.groupby('PRODUCT_TAG_NAME'):
#    if len(td)>3000:
#        td=pd.DataFrame(random.sample(td.values.tolist(),1200) ,columns=td.columns)
    md=td.values.tolist()
    all_list.extend(md)
    num_last=1000-len(td)
    add_list=[] 
    for v in range(num_last):
        line=random.sample(md,1)[0]        
        sentence=line[1]
        s_list=sentence.split(' ')
        sentence_new=sentence
        line_new=line.copy()
        if len(s_list)>=8:
            idx_1=rns.randint(len(s_list)-2)
            idx_2=idx_1+1
            s_list_new=s_list.copy()
            s_list_new[idx_1]=s_list[idx_2]
            s_list_new[idx_2]=s_list[idx_1]
            sentence_new=' '.join(s_list_new)

        line_new[1]=sentence_new
        all_list.append(line_new)

total_data=pd.DataFrame(all_list,columns=data.columns)

d_dict={}
for v in total_data.groupby('PRODUCT_TAG_NAME'):
    d_dict[v[0]]=v[1]
    
aa=total_data.groupby('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME'].count().to_frame('count')
aa['PRODUCT_TAG_NAME']=aa.index

#
#aa=data.groupby('T1')['PRODUCT_TAG_NAME'].count().to_frame('count')
#aa['T1']=aa.index



data_train,data_val= train_test_split(data,test_size=0.05)
logger.info('data_train//////%s'%data_train.shape[0])
#####################################################
text2feature=Text2Feature(seq_len=tag_args.SEQ_LEN,max_words=tag_args.MAX_WORDS, is_rename_tag=False,is_add_data=False,model_id=tag_args.MODEL_ID)
x_train_padded_seqs,x_val_padded_seqs,y1_train,y1_val,y2_train,y2_val,w_sp_train=text2feature.pipeline_fit_transform(data_train,data_val)

#total_data.groupby('T1')['T1'].count()


model=TextCNN(seq_len=tag_args.SEQ_LEN,max_words=tag_args.MAX_WORDS,\
              vocab_len=len(text2feature.vocab),voc_dim=tag_args.VOC_DIM,\
              num_classes1=text2feature.num_classes_list[0], num_classes2=text2feature.num_classes_list[1] , is_pre_train=False,\
              init_learn_rate=tag_args.INIT_LEARN_RATE,batch_size=tag_args.BATCH_SIZE,epoch_max=tag_args.EPOCH_MAX,\
              drop_out_rate=tag_args.DROP_OUT_RATE,early_stop_count=tag_args.EARLY_STOP_COUNT,model_id=tag_args.MODEL_ID)

model.build_model()
model.train(x_train_padded_seqs, y1_train,y2_train,x_val_padded_seqs, y1_val,y2_val,w_sp_train=None) ##data_train['sample_w'].values



#
#_________________________________________________________________________________________________
#2019-07-04 14:28:42,149 - mylib.model_meorient - INFO - epoch: 1   logs:{'val_loss': 0.25132539031971007, 'val_acc': 0.93231531149879543, 'loss': 1.8401713551582979, 'acc': 0.62570627607788598}   takes time:33.99814558029175
#2019-07-04 14:29:14,222 - mylib.model_meorient - INFO - epoch: 2   logs:{'val_loss': 0.20154040004539922, 'val_acc': 0.94356170024844277, 'loss': 0.37244808573180488, 'acc': 0.90689833320288915}   takes time:32.07283091545105
#2019-07-04 14:29:45,958 - mylib.model_meorient - INFO - epoch: 3   logs:{'val_loss': 0.17788924973610262, 'val_acc': 0.94903012806356712, 'loss': 0.28301481858945665, 'acc': 0.92712293009574276}   takes time:31.73391366004944
#2019-07-04 14:30:17,917 - mylib.model_meorient - INFO - epoch: 4   logs:{'val_loss': 0.17919337257490736, 'val_acc': 0.94923648379629322, 'loss': 0.23674176347749373, 'acc': 0.93624819859266317}   takes time:31.957589149475098
#2019-07-04 14:30:49,799 - mylib.model_meorient - INFO - epoch: 5   logs:{'val_loss': 0.18244722456637857, 'val_acc': 0.94975237321420691, 'loss': 0.2072748460084452, 'acc': 0.94296435271072609}   takes time:31.88131332397461
#
#
#

