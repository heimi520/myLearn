

import warnings
warnings.filterwarnings('ignore')
import sys
from os import path
sys.path.append( path.dirname(path.dirname(path.abspath(__file__))))

from fasttext_config import *
from mylib.model_meorient_esemble import *

import pandas as pd
from sklearn.model_selection import train_test_split

cnnconfig=fastextConfig()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=cnnconfig.GPU_DEVICES # 使用编号为1，2号的GPU 
if cnnconfig.GPU_DEVICES=='-1':
    logger.info('using cpu')
else:
    logger.info('using gpu:%s'%cnnconfig.GPU_DEVICES)



import keras.backend.tensorflow_backend as KTF 
gpu_config=tf.ConfigProto()
gpu_config.gpu_options.per_process_gpu_memory_fraction=0.95 ##gpu memory up limit
gpu_config.gpu_options.allow_growth = True # 设置 GPU 按需增长,不易崩掉
session = tf.Session(config=gpu_config) 
# 设置session 
KTF.set_session(session)

import keras 
#keras.backend.set_floatx('float16')

logger.info('keras floatx:%s'%(keras.backend.floatx()))

import multiprocessing
from multiprocessing import *

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import random



tag_pd=pd.read_csv('../data/tag_class/tag_rename_v4.csv')
tag_dict=tag_pd.set_index('PRODUCT_TAG_NAME_ORIG')['PRODUCT_TAG_NAME'].to_dict()

data_his=pd.read_csv('../data/input/ali_amazon_his.csv')
data_test=pd.read_csv('../data/input/ali_amazon_test.csv')

data=pd.concat([data_his,data_test],axis=0)
data['PRODUCT_TAG_NAME']=data['PRODUCT_TAG_NAME'].map(tag_dict)
data['PRODUCT_TAG_ID']=data['PRODUCT_TAG_NAME'].copy()


 
random.seed(1)

cols_list=['BUYSELL', 'PRODUCT_NAME', 'PRODUCT_TAG_ID','PRODUCT_TAG_NAME','T1','sample_w', 'source']
data=data[cols_list]


other_total=pd.read_csv('../data/input/my_other_v5.csv')
data2=pd.read_csv('../data/input/data_add_v5.csv')
total_data=pd.concat([data2[cols_list],other_total[cols_list]],axis=0)
total_data=total_data[total_data['PRODUCT_NAME'].notnull()]

total_data.info()

total_data['source'].unique()

##
#data_train,data_val= train_test_split(total_data.sample(10000),test_size=0.05)
#logger.info('total_data shape//////%s'%total_data.shape[0])
#text2feature=Text2Feature(cnnconfig)
#x_train_padded_seqs,x_val_padded_seqs,y1_train,y1_val,y2_train,y2_val,w_sp_train=text2feature.pipeline_fit_transform(data_train,data_val)
#
#cnnconfig.NUM_TAG_CLASSES=text2feature.num_classes_list[1]
#cnnconfig.TOKENIZER=text2feature.tokenizer 
#cnnconfig.CUT_LIST=text2feature.cut_list
# 
#model=TextCNN(cnnconfig)
#    
#model.build_model()
#model.train(x_train_padded_seqs, [y1_train,y2_train],x_val_padded_seqs, [y1_val,y2_val],w_sp_train=None) ##data_train['sample_w'].values
#
