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


#data=pd.read_excel('../data/check/7.1部分检查：meorient_tag_pred_0628.xlsx')

data=data[data['PRODUCT_NAME'].notnull()]
data=data[data['PRODUCT_NAME'].apply(lambda x:len(x.replace(' ','').replace('\n',''))>=3)]

"""dropduplicate"""
data=data.groupby(['PRODUCT_NAME']).head(1).reset_index() 


detectpipline=DetectPipLine(seq_len=de_args.SEQ_LEN,max_words=de_args.MAX_WORDS, voc_dim=de_args.VOC_DIM, \
                            is_rename_tag=False,model_id=de_args.MODEL_ID,is_pre_train=False,\
                      init_learn_rate=de_args.INIT_LEARN_RATE,batch_size=de_args.BATCH_SIZE,epoch_max=de_args.EPOCH_MAX,\
                      drop_out_rate=de_args.DROP_OUT_RATE,early_stop_count=de_args.EARLY_STOP_COUNT)

lang_pred=detectpipline.predict(data)


data['lang_pred']=lang_pred


#data['PRODUCT_TAG_NAME'].unique()

small_data=data[data['lang_pred']!='en']
en_data=data[data['lang_pred']=='en']
en_data['langfrom']='en'
en_data['langto']='en'
en_data['PRODUCT_NAME_ORIG']=en_data['PRODUCT_NAME'].copy()

print('small data shape',small_data.shape)
#


IS_TRANS_LANG=False


import time
from MyGoogleTrans import *


tolang_list=['ar','es','pl','ru','tr']
    
if IS_TRANS_LANG:
    js=MyGoogleTransTools()
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

d_dict={}
for fromlang in tolang_list:
    ##########################################################################
    subdata=small_data.loc[small_data['lang_pred']==fromlang, ['PRODUCT_NAME','PRODUCT_TAG_NAME','PRODUCT_TAG_ID','COUNTRY_NAME','COUNTRY_ID','BUYSELL']].copy()
    subdata.index=range(len(subdata))
    
    ret_trans_pd=pd.read_csv('../data/input_lang/class_train_data_trans_%s.csv'%fromlang)
    d_dict[fromlang]=ret_trans_pd
    
    ret_trans_pd=pd.merge(ret_trans_pd,subdata[['BUYSELL']],left_index=True,right_index=True,how='left')
    trans_list.append(ret_trans_pd)
    
    
    
trans_pd=pd.concat(trans_list,axis=0)


cols_list=['PRODUCT_NAME_ORIG','PRODUCT_NAME', 'langfrom', 'langto',
       'PRODUCT_TAG_NAME', 'PRODUCT_TAG_ID', 'BUYSELL']

all_data=pd.concat([en_data[cols_list],trans_pd[cols_list]],axis=0)

all_data=all_data[['PRODUCT_NAME_ORIG', 'PRODUCT_NAME', 'langfrom', 'langto',
       'BUYSELL']]

all_data=all_data.rename(columns={'PRODUCT_TAG_NAME':'SYS_TAG'})

all_data.to_csv('../data/input/trans_sample_data.csv',index=False)



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



data_test=all_data


def create_conv_pool(embed,kernel_sizes,filter_num,seq_len):
    """
    """
    lay_list=[]
    for ks in kernel_sizes:
        conv = keras.layers.Conv1D(filters=filter_num, kernel_size=ks,\
                                   strides=1,padding='valid',activation='relu',\
                                   use_bias=True)(embed)  
        pool = keras.layers.MaxPooling1D(pool_size=seq_len-ks+1,strides=1,padding='valid')(conv) 
        pool = keras.layers.Flatten()(pool)
        lay_list.append(pool)
    return lay_list

 

def generate_batch_data(x, y, batch_size,is_shuffle=True):
    """逐步提取batch数据到显存，降低对显存的占用"""
    batches = (len(y) + batch_size - 1)//batch_size
    if is_shuffle:
        idx=list(range(len(y)))
        rns.shuffle(idx)
        x=x[idx]
        y=y[idx]
    
    cnt=0
    for i in range(batches):
        cnt+=1
        X = x[i*batch_size : (i+1)*batch_size]
        Y = y[i*batch_size : (i+1)*batch_size]
        yield (X, Y)




def build():
    tf.reset_default_graph()    
    
    kernel_sizes= [2,3,4,5,6,7]
    filter_num=100
    
    seq_len=tag_args.SEQ_LEN
    max_words=tag_args.MAX_WORDS
    voc_dim=tag_args.VOC_DIM
    drop_out_rate=tag_args.DROP_OUT_RATE
    
    num_classes=240####text2feature.num_classes
    init_learn_rate=tag_args.INIT_LEARN_RATE
    
    place_x= Input(shape=(seq_len,))
    place_y=Input(shape=(num_classes,))
    
    is_pre_train=False
    #place_x= Input(shape=(self.seq_len,))
    if is_pre_train:
    #    embed_w=model.get_layer('embedding_1').get_weights()[0]
    #    np.savetxt('../data/input/embed.csv',embed_w)
        embed_w=np.loadtxt('../data/input/embed.csv','float32')
    embed = keras.layers.Embedding(max_words if max_words is not None else vocab_len + 1, ###self.vocab_len + 1,
                                   voc_dim ,
                                   trainable=True if not is_pre_train  else False,
                                   weights=[embed_w] if is_pre_train else None , 
                                   input_length=seq_len)(place_x)
    
    #
    lay_list=create_conv_pool(embed,kernel_sizes,filter_num,seq_len)
    net = keras.layers.concatenate(lay_list, axis=1)
    
    net =Dropout(drop_out_rate)(net)
    net= Dense(256, activation='relu')(net)
    net =Dropout(drop_out_rate)(net)
    
    softmax_before=Dense(num_classes)(net)
    prob = tf.nn.softmax(softmax_before)
    
    pred_label=tf.argmax(prob,axis=1)
    true_label=tf.argmax(place_y,axis=1)
        

    acc=tf.divide(tf.reduce_sum( tf.cast(tf.equal(pred_label,true_label) ,tf.float32 )) ,tf.reduce_sum(tf.ones_like(true_label,dtype=tf.float32 )) )
    
    model = keras.models.Model(inputs=place_x, outputs=softmax_before)
    model.summary()
    
    cross_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=place_y,logits=softmax_before))
    #                          tf.nn.softmax_cross_entropy_with_logits
    
    #train_optim = tf.train.AdamOptimizer().minimize(cross_loss)
     	 
    optimizer = tf.train.AdamOptimizer(init_learn_rate)
    train_op=optimizer.minimize(cross_loss)
    

    return place_x,place_y,prob,pred_label,true_label,acc,train_op,cross_loss,softmax_before



def pipeline_predict(line_list):
    col=pd.DataFrame(line_list,columns=['PRODUCT_NAME'])  
    text2feature=Text2Feature(seq_len=tag_args.SEQ_LEN,max_words=tag_args.MAX_WORDS,is_rename_tag=False,model_id=tag_args.MODEL_ID)
    x_test_padded_seqs=text2feature.pipeline_transform(col)

#    data_test['label']=text2feature.tag2label(data_test['PRODUCT_TAG_ID'])
 
    place_x,place_y, prob,pred_label,true_label,acc,train_op,cross_loss,softmax_before=build()

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_' in g.name]
    var_list += bn_moving_vars
    saver=tf.train.Saver(var_list,max_to_keep=3)##keep latest N model
    
    
    import keras.backend.tensorflow_backend as KTF 
    gpu_config=tf.ConfigProto()
    gpu_config.gpu_options.per_process_gpu_memory_fraction=0.95 ##gpu memory up limit
    gpu_config.gpu_options.allow_growth = True # 设置 GPU 按需增长,不易崩掉
    
    ####session init######
    sess = tf.Session(config=gpu_config)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())   
    #self.sess=sess
    
    model_path='../data/temp/tf_model' 
    model_name='mytest'
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_dir=model_path)) 
    [tag_max,y_prob]=sess.run([pred_label,prob],feed_dict={place_x:x_test_padded_seqs})  

    
    #y_prob=model.predict(x_test_padded_seqs)
    
    tag_max_int=np.argmax(y_prob,axis=1)
    tag_max=text2feature.num2label(tag_max_int)
    prob_max=np.max(y_prob,axis=1)
    
    y_prob2=y_prob.copy()
    for k,v in enumerate(tag_max_int):
        y_prob2[k,v]=0
    
    
    tag_second_int=np.argmax(y_prob2,axis=1)
    tag_second=text2feature.num2label(tag_second_int)
    
    prob_second=np.max(y_prob2,axis=1)
 
    return tag_max,tag_second,prob_max,prob_second
#





line_list=data_test['PRODUCT_NAME'].tolist()
tag_max,tag_second,prob_max,prob_second=pipeline_predict(line_list)
data_test['TAG_NAME_PRED1']=tag_max
data_test['TAG_NAME_PRED2']=tag_second
data_test['prob_max']=prob_max
data_test['prob_second']=prob_second
data_test['second2max']=data_test['prob_second']/data_test['prob_max']
#data_test['TAG_NAME_PRED']=data_test['TAG_NAME_PRED1'].copy()
data_test['TAG_NAME_PRED']=np.where(data_test['prob_max']>=0.1,data_test['TAG_NAME_PRED1'],'Other')
#data_test['TAG_NAME_PRED']=np.where(  (data_test['TAG_NAME_PRED2']=='Other' )|(data_test['TAG_NAME_PRED1']=='Other') ,'Other',data_test['TAG_NAME_PRED1'])



#data_test['prob_max'].hist(bins=100)

#data_test.to_csv('../data/output/meorient_tag_pred_0628.csv',index=False,encoding='utf-8')
#

#data_test.to_csv('../data/output/meorient_tag_pred_rnn0701.csv',index=False,encoding='utf-8')

#data_test.to_csv('../data/output/meorient_tag_pred_0703.csv',index=False,encoding='utf-8')

data_test.to_csv('../data/output/meorient_tag_pred_0703_v4.csv',index=False,encoding='utf-8')


aa= data_test.groupby('TAG_NAME_PRED')['TAG_NAME_PRED'].count().sort_values().to_frame('count')
aa['TAG_NAME_PRED']=aa.index



#cd.to_csv('../data/output/tag_stat_0627.csv')

#
#data_test['TAG_NAME_PRED0']=data_test['TAG_NAME_PRED'].copy()
#idx=data_test['prob_max']<0.2
#data_test.loc[idx, 'TAG_NAME_PRED']='other'

#data_test['prob_max'].hist(bins=100)



#text2feature=Text2Feature(seq_len=tag_args.SEQ_LEN,max_words=tag_args.MAX_WORDS,is_rename_tag=False,model_id=tag_args.MODEL_ID)
#x_test_padded_seqs=text2feature.pipeline_transform(data_test)
#
#
#
#model=TextCNN(max_words=tag_args.MAX_WORDS, model_id=tag_args.MODEL_ID)
#
#
#y_pred_class=model.predict_classes(x_test_padded_seqs)
#y_pred=text2feature.num2label(y_pred_class)
#
#data_test['TAG_NAME_PRED']=text2feature.label2tagname(y_pred)
#
aa=data_test[['PRODUCT_NAME_ORIG','PRODUCT_NAME','TAG_NAME_PRED','TAG_NAME_PRED2', 'prob_max', 'prob_second']]

pred_dict={}
for v in aa.groupby('TAG_NAME_PRED'):
    pred_dict[v[0]]=v[1]




cd=aa.groupby('TAG_NAME_PRED')['TAG_NAME_PRED'].count().sort_values(ascending=False).to_frame('count')
cd['tag']=cd.index
cd['pct']=cd['count']/cd['count'].sum()
#cd.to_csv('../data/output/tag_weight.csv',index=False)



pred_list=[]
for v in cd.index:
    pred_list.append(aa[aa['TAG_NAME_PRED']==v])








