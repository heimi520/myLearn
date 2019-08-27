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
from mylib.model_meorient import *


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

data=pd.read_csv('../data/input/all_data_dp.csv')

data=data[data['PRODUCT_NAME'].notnull()]
data=data[data['PRODUCT_NAME'].apply(lambda x:len(x.replace(' ','').replace('\n',''))>=3)]

"""dropduplicate"""
data=data.groupby(['PRODUCT_NAME']).head(1).reset_index() 



detectpipline=DetectPipLine(seq_len=de_args.SEQ_LEN,is_rename_tag=False,model_id=de_args.MODEL_ID)
lang_pred=detectpipline.predict(data[['PRODUCT_NAME']])

data['lang_pred']=lang_pred

small_data=data[data['lang_pred']!='en']
en_data=data[data['lang_pred']=='en']
en_data['langfrom']='en'
en_data['langto']='en'
en_data['PRODUCT_NAME_ORIG']=en_data['PRODUCT_NAME'].copy()

print('small data shape',small_data.shape)
#
import time
from MyGoogleTrans import *

js=MyGoogleTransTools()

tolang_list=['ar','es','pl','ru','tr']

for fromlang in tolang_list:
    ##########################################################################
    subdata=small_data.loc[small_data['lang_pred']==fromlang, ['PRODUCT_NAME','PRODUCT_TAG_NAME','PRODUCT_TAG_ID','COUNTRY_NAME','COUNTRY_ID']].copy()
    subdata.index=range(len(subdata))
    batch_idx_list, batch_line_list,batch_str_list=to_batch_data(subdata)
    batch_trans_pd=trans_batch(js,fromlang,'en',batch_idx_list,batch_line_list,batch_str_list)
    
    ret_trans_pd=pd.merge(batch_trans_pd,subdata[['PRODUCT_TAG_NAME','PRODUCT_TAG_ID']],left_on=['idx'],right_index=True,how='left')
    ret_trans_pd=ret_trans_pd.rename(columns={'trans_text':'PRODUCT_NAME','source_text':'PRODUCT_NAME_ORIG'})
    ret_trans_pd.to_csv('../data/input_lang/class_train_data_trans_%s.csv'%fromlang,index=False)
  
 

trans_list=[]
for fromlang in tolang_list:
    ##########################################################################
    subdata=small_data.loc[small_data['lang_pred']==fromlang, ['PRODUCT_NAME','PRODUCT_TAG_NAME','PRODUCT_TAG_ID','COUNTRY_NAME','COUNTRY_ID','BUYSELL']].copy()
    subdata.index=range(len(subdata))
    
    ret_trans_pd=pd.read_csv('../data/input_lang/class_train_data_trans_%s.csv'%fromlang)
    
    ret_trans_pd=pd.merge(ret_trans_pd,subdata[['BUYSELL']],left_index=True,right_index=True,how='left')
    trans_list.append(ret_trans_pd)
trans_pd=pd.concat(trans_list,axis=0)
trans_pd['PRODUCT_NAME']=np.where(trans_pd['PRODUCT_NAME'].isnull(),trans_pd['PRODUCT_NAME_ORIG'],trans_pd['PRODUCT_NAME'])


cols_list=['PRODUCT_NAME_ORIG','PRODUCT_NAME', 'langfrom', 'langto',
       'PRODUCT_TAG_NAME', 'PRODUCT_TAG_ID', 'BUYSELL']

all_data=pd.concat([en_data[cols_list],trans_pd[cols_list]],axis=0)
all_data.to_csv('../data/input/trans_sample_data.csv',index=False)







#sentence='---'.join(small_data['PRODUCT_NAME'].tolist()[:30])
#
#js=MyGoogleTransTools()
##sentence='good'
#
#js=MyGoogleTransTools()
#all_trans_list=[]
#for v in small_data[['PRODUCT_NAME','lang_pred']].groupby('lang_pred'):
#    fromlang=v[0]
#    subdata=v[1]
#    batch_line_list,batch_str_list=to_batch_data(subdata)
#    batch_trans_pd=trans_batch(fromlang,batch_line_list,batch_str_list)
#    all_trans_list.append(batch_trans_pd)
#    
 




#sentence2='Koszula meska---koszule---odziez damska---playeras---Pantalones cortos bermudas de Bermudas para hombre cómodos de la venta caliente diversos en verano---Playeras---Nuevo estilo 3 en 1 Forma de cuchillo suizo Cable de datos USB multifuncional Línea de datos USB, Micro USB, tipo C, 8 pines con luz LED para todos los productos digitales---Üst Satmak Usams USB Veri Kablosu Düz Kablo Şarj Kablosu---UENJOY - SERISE TRINKLE / Cable Nylon trenzado antitangular Tipo C Cable 3FT / 1M A 2.0 a USB-C Cargador rápido Dispositivos USB C---PANTALON MILITAR,GORRAS---koszulki---Podkoszulki prostujące kręgosłup i postawę .Men s body shaper vest---odziez wojskowa---Cable magnético micro del cable USB del LED Cable trenzado de nylon del cargador de 90 grados de la forma de L---D-Line2 Cable Micro USB Voltaje y corriente Cable de sincronización de datos USB para Samsung Xiaomi Huawei Microusb Cable---Lace Fabric & Embroidery---Chamarras para Niños---Cómputo y telefonia---Elecronica y computo---Electrónicos y computo---sukienki dla dzieci---Lencería---Cable de micrófono profesional---Conectores de audio y cables de audio---Ropa Fajas, corset, deportiva---ropa interior de mujer---İç giyim'
#sentence2='Koszula meska'
#text2=js.translate(sentence2)

#import time
#from GoogleTransJS import *
#
#js=Py4Js()
#t0=time.time()
#ret_list=[]
#for k, v in enumerate(small_data[['lang_pred','PRODUCT_NAME','PRODUCT_TAG_NAME', 'PRODUCT_TAG_ID']].values):
#    lang=v[0]
#    line=v[1]
#    tag_id=v[2]
#    tag_name=v[3]
#    t1=time.time()
#    text=js.translate(line)
#    ret_list.append([lang,line,text,tag_id,tag_name])
##    res=js.translate(line)
#    t2=time.time()
#    time.sleep(0.5)
#    print('*************************************************')
#    print('lang/////////',lang)
#    print('source    ///',line)
#    print('dest text //',text)
#    print('k//',k,'takes time//',t2-t1,'time longs ///',t2-t0)
#
#

#ret_pd=pd.DataFrame(ret_list,columns=['lang_pred','PRODUCT_NAME','PRODUCT_NAME_TRANS','PRODUCT_TAG_NAME','PRODUCT_TAG_ID'])
#ret_pd.to_csv('../data/input/trans_sample_data.csv',index=False)

#









