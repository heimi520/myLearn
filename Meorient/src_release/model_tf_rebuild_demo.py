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


import pandas as pd
from sklearn.model_selection import train_test_split


class TagModelArgs():
    GPU_DEVICES='0' ##-1:CPU        
    SEQ_LEN=40 #100
    MAX_WORDS=10000 ##10000 #5000
    VOC_DIM=100 ##100
    BATCH_SIZE=256##64 ###64
    INIT_LEARN_RATE=0.001
    EPOCH_MAX=50
    DROP_OUT_RATE=0.5  ###0.3
    EARLY_STOP_COUNT=6
    
    ########################################################################################
#    MODEL_ID='ali_amazon_cnn_win123456_3c_parral_data_clean_add_newtag_Ethnic_Clothing500' # base model real one
    MODEL_ID='ali_amazon_cnn_win123456_tf' # base model real one





tag_args=TagModelArgs()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=tag_args.GPU_DEVICES # 使用编号为1，2号的GPU 
if tag_args.GPU_DEVICES=='-1':
    logger.info('using cpu')
else:
    logger.info('using gpu:%s'%tag_args.GPU_DEVICES)


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

data=data[['PRODUCT_TAG_NAME','PRODUCT_TAG_ID','PRODUCT_NAME','source','BUYSELL']]
data['sample_w']=1


def get_other_data():
    from nltk.corpus import gutenberg
    import random
    line_list=[]
    for field in gutenberg.fileids()[:2]:
        for v in gutenberg.open(field):
            line_list.append(v)
    line_list=[v for v in line_list if len(re.sub('(^\s*)|(\s*$)','',v).split(' '))>5]
    line_list=random.sample(line_list,2000)
    other=pd.DataFrame( line_list,columns=['PRODUCT_NAME'])
    other['sect_flag']=1
    other['BUYSELL']='sell'
    other['source']='other'
    other['PRODUCT_TAG_ID']='Other'
    other['PRODUCT_TAG_NAME']='Other'
    other['sample_w']=1
    
    return other



#other=pd.read_csv('../data/input/my_other.csv')


other=get_other_data()
data=pd.concat([data,other],axis=0)

#data=data[data['PRODUCT_TAG_NAME']!='Ethnic Clothing']

cols_list=['BUYSELL', 'PRODUCT_NAME', 'PRODUCT_TAG_ID', 'PRODUCT_TAG_NAME','sample_w', 'source']
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
    num_last=500-len(td)
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

    
data_train,data_val= train_test_split(total_data,test_size=0.05)
#####################################################
text2feature=Text2Feature(seq_len=tag_args.SEQ_LEN,max_words=tag_args.MAX_WORDS, is_rename_tag=False,is_add_data=False,model_id=tag_args.MODEL_ID)
x_train_padded_seqs,x_val_padded_seqs,y_train,y_val,w_sp_train=text2feature.pipeline_fit_transform(data_train,data_val)


def create_conv_pool(embed,kernel_sizes,filter_num,seq_len):
    """
    """
    lay_list=[]
    for ks in kernel_sizes:
        conv = tf.keras.layers.Conv1D(filters=filter_num, kernel_size=ks,\
                                   strides=1,padding='valid',activation='relu',\
                                   use_bias=True)(embed)  
        pool = tf.keras.layers.MaxPooling1D(pool_size=seq_len-ks+1,strides=1,padding='valid')(conv) 
        pool = tf.keras.layers.Flatten()(pool)
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
    
    place_x= tf.keras.Input(shape=(seq_len,))
    place_y= tf.keras.Input(shape=(num_classes,))
    
    is_pre_train=False
    #place_x= Input(shape=(self.seq_len,))
    if is_pre_train:
    #    embed_w=model.get_layer('embedding_1').get_weights()[0]
    #    np.savetxt('../data/input/embed.csv',embed_w)
        embed_w=np.loadtxt('../data/input/embed.csv','float32')
    embed = tf.keras.layers.Embedding(max_words if max_words is not None else vocab_len + 1, ###self.vocab_len + 1,
                                   voc_dim ,
                                   trainable=True if not is_pre_train  else False,
                                   weights=[embed_w] if is_pre_train else None , 
                                   input_length=seq_len)(place_x)
    
    #
    lay_list=create_conv_pool(embed,kernel_sizes,filter_num,seq_len)
    net = tf.keras.layers.concatenate(lay_list, axis=1)
    
    net =tf.keras.layers.Dropout(drop_out_rate)(net)
    net= tf.keras.layers.Dense(256, activation='relu')(net)
    net =tf.keras.layers.Dropout(drop_out_rate)(net)
    
    softmax_before=tf.keras.layers.Dense(num_classes)(net)
    prob = tf.nn.softmax(softmax_before)
    
    pred_label=tf.argmax(prob,axis=1)
    true_label=tf.argmax(place_y,axis=1)
        

    acc=tf.divide(tf.reduce_sum( tf.cast(tf.equal(pred_label,true_label) ,tf.float32 )) ,tf.reduce_sum(tf.ones_like(true_label,dtype=tf.float32 )) )
    
    model = tf.keras.models.Model(inputs=place_x, outputs=softmax_before)
    model.summary()
    
    cross_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=place_y,logits=softmax_before))
    #                          tf.nn.softmax_cross_entropy_with_logits
    
    #train_optim = tf.train.AdamOptimizer().minimize(cross_loss)
     	 
    optimizer = tf.train.AdamOptimizer(init_learn_rate)
    train_op=optimizer.minimize(cross_loss)
    

    return place_x,place_y,prob,pred_label,true_label,acc,train_op,cross_loss,softmax_before


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


epoch_max=tag_args.EPOCH_MAX
early_stop_count=tag_args.EARLY_STOP_COUNT

best_metrics_val=np.inf
g_stop_flag=False
early_count=0

batch_size=tag_args.BATCH_SIZE

model_path='../data/temp/tf_model' 
model_name='mytest'

import gc
loss_list=[]
####################
t0=time.time()
for i in range(epoch_max):
    loss_train=0
    acc_train=0
    cnt=0
    t1=time.time()
    for k, [batch_x,batch_y] in enumerate(generate_batch_data(x_train_padded_seqs, y_train, batch_size,is_shuffle=True)):
#        print(k)
        [_,loss_train_,acc_train_ ,score_,prob_]=sess.run([train_op,cross_loss,acc,softmax_before,prob],\
        feed_dict={place_x:batch_x,place_y:batch_y})
#        print(loss_train_)
        loss_train+=loss_train_
        acc_train+=acc_train_
        cnt+=1
    loss_train=loss_train/cnt
    acc_train=acc_train/cnt
    t2=time.time()
#    print('epoch//',i,'loss train',loss_train)
#    
    loss_val=0
    acc_val=0
    cnt=0
    for k, [batch_x,batch_y] in enumerate(generate_batch_data(x_val_padded_seqs, y_val, batch_size,is_shuffle=False)):
        [loss_val_,acc_val_]=sess.run([cross_loss,acc],\
        feed_dict={place_x:batch_x,place_y:batch_y})
        loss_val+=loss_val_
        acc_val+=acc_val_
        cnt+=1
    loss_val=loss_val/cnt
    acc_val=acc_val/cnt
    
    print('epoch',i,'cost_train',loss_train,'loss_val',loss_val,'acc_train',acc_train,'acc_val',acc_val, 'takes time',t2-t1,'time longs',t2-t0)
    gc.collect()
    loss_list.append([i,loss_train,loss_val])

#    metrics_val=loss_val
    metrics_val=-acc_val
    ###提前停止，metrics_val不再减小时停止################################
    if metrics_val<best_metrics_val:
        best_metrics_val=metrics_val
        early_count=0####更新最小loss_val时，重置计数器
        saver.save(sess,'%s/textcnn_%s.ckpt'%(model_path, model_name),global_step=i) ##save best model
    else:
        early_count+=1
        print('early count//////////////',early_count)
    if early_count>=early_stop_count:
        ###停止前保存模型
        g_stop_flag=True
        print('early stop,best epoch//',i,' k///',k)
        break
        
    if g_stop_flag:
        break
  
    
    
    
    
    






