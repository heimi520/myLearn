#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 22:38:15 2019

@author: heimi
"""



import warnings
warnings.filterwarnings('ignore')

import multiprocessing
from multiprocessing import *

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import keras.layers as layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models  import *
from keras.layers import *
import keras
from keras import *
from keras.models import load_model


from sklearn.preprocessing import LabelEncoder

import tensorflow as tf

import re
from string import punctuation
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


from sklearn.preprocessing import LabelEncoder
from sklearn import *
from numba import jit
import os
import time
import pickle
import collections
##################################################
from keras.callbacks import ModelCheckpoint, TensorBoard,EarlyStopping, BaseLogger
import numpy as np
import pandas as pd
import os
from numpy.random import seed
import random
random.seed(1)
os.environ['PYTHONHASHSEED'] = '-1'
seed(0)  
rns=np.random.RandomState(0)
tf.set_random_seed(1)
tf.reset_default_graph()

import logging
logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


data_dir='../data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
data_input_dir='../data/input'
if not os.path.exists(data_input_dir):
    os.makedirs(data_input_dir)
  
data_output_dir='../data/output'
if not os.path.exists(data_output_dir):
    os.makedirs(data_output_dir)    

data_temp_dir='../data/temp'  
if not os.path.exists(data_temp_dir):
    os.makedirs(data_temp_dir)  








class multiProcess(Process):
    def __init__(self,data,worker):
        super().__init__()
        self.col=data
        self.worker=worker
#        
    def set_worder(self,func):
        self.worker=func
        
    def set_tobatch(self,func):
        self.to_batch_data=func
    
    
        
    def run(self):
        col=self.col
        num=cpu_count()
        batch_dict=self.to_batch_data(col,num)
        
        stemer = PorterStemmer()
        
        manager = Manager()
        return_dict = manager.dict()
        jobs = []
    
        for (k,line_list) in batch_dict.items():
            logger.info('start worker//%s'%k)
            p = multiprocessing.Process(target=self.worker, args=(stemer,k,line_list,return_dict))
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
            



class TrainingMonitor(BaseLogger):
    def __init__(self):
		# 保存loss图片到指定路径，同时也保存json文件
        super(TrainingMonitor, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        self.t1=time.time()
        self.seen = 0
        self.totals = {}
        

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        batch_size = logs.get('size', 0)
        self.seen += batch_size
        for k, v in logs.items():
            if k in self.totals:
                self.totals[k] += v * batch_size
            else:
                self.totals[k] = v * batch_size

    def on_epoch_end(self, epoch, logs=None):
        self.t2=time.time()
        logger.info('epoch: %s   logs:%s   takes time:%s'%(epoch+1,logs,self.t2-self.t1))
        if logs is not None:
            for k in self.params['metrics']:
                if k in self.totals:
                    # Make value available to next callbacks.
                    logs[k] = self.totals[k] / self.seen




class TextCNN(object):
    
    model_net_filename='textcnn_net'
    model_w_filename='textcnn_w'
    
    def __init__(self,seq_len=None,max_words=None,vocab_len=None,voc_dim=None,num_classes=None,is_pre_train=False,\
                 init_learn_rate=0.001,batch_size=512,epoch_max=50,drop_out_rate=0.5,early_stop_count=5, model_id='base_model'):
        
        self.kernel_sizes =[1,1,2,2,4,6] ##[2,3,4,5,6,7] ##[2,3,4,6,8] ##[2,3,4,5,6] ### [2,2,3,3,4,4]
        self.filter_num=100
        self.max_words=max_words
        self.seq_len=seq_len
        self.vocab_len=vocab_len
        self.voc_dim=voc_dim
        self.is_pre_train=is_pre_train
        self.num_classes=num_classes
        self.drop_out_rate=drop_out_rate
        self.init_learn_rate=init_learn_rate
        self.early_stop_count=early_stop_count
        
        self.batch_size=batch_size
        self.epoch_max=epoch_max
        
        self.data_model_dir=os.path.join('../data/temp' ,model_id) 
        if not os.path.isdir(self.data_model_dir):
            os.makedirs(self.data_model_dir)  
        
        
        logger.info( ("TextCNN  __init__ params:,max_words:%s,seq_len=%s,vocab_len=%s, voc_dim=%s, num_classes=%s, is_pre_train=%s,\
                 init_learn_rate=%s,batch_size=%s,epoch_max=%s,drop_out_rate=%s, early_stop_count=%s,model_id=%s"%(\
                 max_words,seq_len,vocab_len,voc_dim,num_classes,is_pre_train,init_learn_rate,batch_size,epoch_max,\
                 drop_out_rate,early_stop_count,model_id)).replace(' ','').replace(',','\n') )
        
        

    def build_model(self):
        """
        """
        ##textcnn####conv1
        place_x= Input(shape=(self.seq_len,))
        if self.is_pre_train:
        #    embed_w=model.get_layer('embedding_1').get_weights()[0]
        #    np.savetxt('../data/input/embed.csv',embed_w)
            embed_w=np.loadtxt('../data/input/embed.csv','float32')
        embed = keras.layers.Embedding(self.max_words if self.max_words is not None else self.vocab_len + 1, ###self.vocab_len + 1,
                                       self.voc_dim ,
                                       trainable=True if not self.is_pre_train  else False,
                                       weights=[embed_w] if self.is_pre_train else None , 
                                       input_length=self.seq_len)(place_x)
        
        lay_list=self.create_conv_pool(embed,self.kernel_sizes,self.filter_num,self.seq_len)
        net = keras.layers.concatenate(lay_list, axis=1)
        
        
#        net = keras.layers.concatenate(lay_list, axis=1)
        
#        net =Dropout(self.drop_out_rate)(net)
#        ###############################################
#        net=GRU(128, dropout=self.drop_out_rate, recurrent_dropout=self.drop_out_rate,return_sequences=False)(embed) 
#        ########################################
#        net=Concatenate(axis=1)([net1,net2])
##        net=Bidirectional(GRU(64, dropout=self.drop_out_rate, recurrent_dropout=self.drop_out_rate,return_sequences=False))(embed) ###if multri rnn then return_sequences=True
#        net =Dropout(self.drop_out_rate)(net)
        
#        net=GRU(128, dropout=0.3, recurrent_dropout=0.3,return_sequences=False)(embed)
        
        net =Dropout(self.drop_out_rate)(net)
        net= Dense(256, activation='relu')(net)
        net =Dropout(self.drop_out_rate)(net)
#        net=Dense(32, activation='relu')(net)
#        
        output=Dense(self.num_classes, activation='softmax')(net)

        self.model = keras.models.Model(inputs=place_x, outputs=output)
        self.model.summary()
        
        opt=keras.optimizers.Adam(lr=self.init_learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08,clipvalue=1.0)
        self.model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=['accuracy'])
        return self.model

        
    def train(self,x_train_padded_seqs, y_train, x_val_padded_seqs, y_val,w_sp_train=None):
        """
        """
        logger.info('training model....')
        self.model.save(os.path.join(self.data_model_dir,'%s.h5'%self.model_net_filename))
        #记录所有训练过程，每隔一定步数记录最大值
        tensorboard = TensorBoard(log_dir=self.data_model_dir)
        checkpoint = ModelCheckpoint(os.path.join(self.data_model_dir,'%s.h5'%self.model_w_filename),
                                     monitor="val_acc",
                                     mode='max',
                                     save_weights_only=True,
                                     save_best_only=True, 
                                     verbose=0,###no logs
                                     period=1)
         
                 
        monitor=TrainingMonitor()
        earlystop=EarlyStopping(monitor='val_acc', patience=self.early_stop_count,mode='max', verbose=2)
        callback_lists=[tensorboard,checkpoint,earlystop,monitor]
        ##########################################################
#        self.history=self.model.fit(
#                                    x_train_padded_seqs, y_train, 
#                                    batch_size=self.batch_size,              
#                                    initial_epoch=0,
#                                    epochs=self.epoch_max, 
#                                    validation_data = (x_val_padded_seqs, y_val),
#                                    verbose=0,
#                                    callbacks=callback_lists,
#                                    sample_weight=w_sp_train,
#                                    
##                                        workers=4,
##                                        use_multiprocessing=True,
#                                    )
        ##############################################################################
        self.history=self.model.fit_generator(
                                        self.generate_batch_data_random(x_train_padded_seqs, y_train, self.batch_size),                                                      
                                        steps_per_epoch=len(x_train_padded_seqs)//self.batch_size,
                                        initial_epoch=0,
                                        nb_epoch=self.epoch_max, 
                                        validation_data = (x_val_padded_seqs, y_val),
                                        verbose=0,
                                        callbacks=callback_lists,
#                                        sample_weight=w_sp_train,
                                        
#                                        workers=4,
#                                        use_multiprocessing=True,
                                        )
        return self.history

#
#    def partical_predict(self,x_test_padded_seqs):
#        self.model=load_model(os.path.join(self.data_model_dir,'%s.h5'%self.model_net_filename))   # H    
#        self.model.load_weights(os.path.join(self.data_model_dir,'%s.h5'%self.model_w_filename))
#        
#        return self.model.predict(x_test_padded_seqs) 
#    
            
    def predict(self,x_test_padded_seqs):
        self.model=load_model(os.path.join(self.data_model_dir,'%s.h5'%self.model_net_filename))   # H    
        self.model.load_weights(os.path.join(self.data_model_dir,'%s.h5'%self.model_w_filename))
        return self.model.predict(x_test_padded_seqs,batch_size=256) 
    
    def predict_classes(self,x_test_padded_seqs):
        prob=self.predict(x_test_padded_seqs)
        y_pred_class=np.argmax(prob,axis=1)
        return y_pred_class 
    
    
    def plot_history(self):
        # 绘制训练 & 验证的准确率值
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        
        # 绘制训练 & 验证的损失值
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()


#    
    def create_conv_pool(self,embed,kernel_sizes,filter_num,seq_len):
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


    
        
    
    def generate_batch_data_random(self,x, y, batch_size):
        """逐步提取batch数据到显存，降低对显存的占用"""
        batches = (len(y) + batch_size - 1)//batch_size
        while(True):
            cnt=0
            idx=list(range(len(y)))
            rns.shuffle(idx)
            x=x[idx]
            y=y[idx]
            
            idx_col=list(range(x.shape[1]))
            for i in range(batches):
                cnt+=1
                X = x[i*batch_size : (i+1)*batch_size]
                Y = y[i*batch_size : (i+1)*batch_size]
                
#                rns.shuffle(idx_col)
#                X=X[:,idx_col]
                
                yield (X, Y)
    


class Text2Feature(object):
    token_obj_name='token_obj'
    tag_id2name_dict_name='tag_id2name_dict'
    tag2label_dict_name='tag2label_dict'
    label2num_dict_name='label2num_dict'
#    cols_list=['PRODUCT_NAME','CONTENT']
#    cols_list=['PRODUCT_NAME','COUNTRY_NAME','T3','T4']
    cols_list=['PRODUCT_NAME']
    is_use_char=False
    
    use_multi_process=True  
    
    def __init__(self,seq_len=150,max_words=20000,is_rename_tag=True,is_add_data=False,model_id='base_model'):
        self.seq_len=seq_len
        self.max_words=max_words
        self.is_rename_tag=is_rename_tag
        self.is_add_data=is_add_data
        
        self.data_model_dir=os.path.join('../data/temp' ,model_id) 
        if not os.path.isdir(self.data_model_dir):
            os.makedirs(self.data_model_dir)  
        
        logger.info(('Text2Feature __init__ params:,seq_len=%s,max_words:%s,is_rename_tag=%s,\
                    is_add_data=%s, model_id=%s'%(seq_len,max_words,is_rename_tag,is_add_data,model_id)).replace(' ','').replace(',','\n')  )
        
        logger.info('Text2Feature __init__ params: is_use_char:%s'%self.is_use_char)
    def pre_process(self,data_train,data_val):
        data_train['flag']='train'
        data_val['flag']='val'
        data=pd.concat([data_train,data_val],axis=0)   
        
        tag_id2name_dict=data.set_index('PRODUCT_TAG_ID')['PRODUCT_TAG_NAME'].to_dict()
        self.save_obj(tag_id2name_dict,self.tag_id2name_dict_name)
        
        data['text']=self.str_col_cat(data,self.cols_list)
        
        return data
    
    def pipeline_fit_transform(self,data_train,data_val):
        
        data=self.pre_process(data_train,data_val)
        data['text']=self.text_clean(data['text'])
        return self.fit_transform(data)
     
    def pipeline_transform(self,data_test):
        data_test['text']=self.str_col_cat(data_test,self.cols_list)
        data_test['text']=self.text_clean(data_test['text']) 
        x_test=data_test['text']
        tokenizer=self.read_obj(self.token_obj_name)   
        x_test_word_ids=tokenizer.texts_to_sequences(x_test) 
        x_test_padded_seqs=pad_sequences(x_test_word_ids, maxlen=self.seq_len)
        return x_test_padded_seqs
    

    def fit_transform(self,data):
        """
        """
        t1=time.time()
#        data_train=self.train_data_boost(data[data['flag']=='train'],is_add=self.is_add_data)
        data=pd.concat([data,data.loc[data['flag']=='val', ['PRODUCT_TAG_ID','text','flag']]],axis=0) 

        tokenizer=Tokenizer(num_words=self.max_words, filters='',lower=True,split=' ')  
        tokenizer.fit_on_texts(data['text'])
        self.save_obj(tokenizer,self.token_obj_name)
        self.vocab=tokenizer.word_index 
        self.tokenizer=tokenizer
    
        ft=data['text']
        word_ids=tokenizer.texts_to_sequences(ft)
        padded_seqs=pad_sequences(word_ids,maxlen=self.seq_len) #将超过固定值的部分截掉，不足的在最前面用0填充
        x_train_padded_seqs,x_val_padded_seqs=padded_seqs[data['flag']!='val'],padded_seqs[data['flag']=='val']

        ##########################################
        tag2label_dict=self.create_tag_dict(data[data['flag']!='add'],self.is_rename_tag)   
        self.save_obj(tag2label_dict,self.tag2label_dict_name)
        data['label']=self.tag2label(data['PRODUCT_TAG_ID'])
        
        uc=list(data['label'].unique())
        label2num_dict={v:k for k,v in enumerate(uc)}
        self.save_obj(label2num_dict,self.label2num_dict_name)
        
        data['y']=data['label'].map(label2num_dict)
        self.num_classes=len(data['y'].unique())
        
        yy=keras.utils.to_categorical(data['y'], num_classes=self.num_classes) 
        y_train,y_val,w_train=yy[data['flag']!='val'],yy[data['flag']=='val'],data.loc[data['flag']!='val','sample_w']
        
        t2=time.time()
        
        logger.info('feature fit_transform takes time///%s'%(t2-t1))
        return x_train_padded_seqs,x_val_padded_seqs,y_train,y_val,w_train


    def tag2label(self,y):
        tag2label_dict=self.read_obj(self.tag2label_dict_name)
        return [tag2label_dict.get(v,'other')  for v in  y]
        
    def num2label(self,pred):
        label2num_dict=self.read_obj(self.label2num_dict_name)
        num2label_dict={v:k for k,v in label2num_dict.items() }
        return [num2label_dict.get(v,'other')  for v in  pred]
    
    def label2tagname(self,label):
        tag_id2name_dict=self.read_obj(self.tag_id2name_dict_name)
        return [tag_id2name_dict.get(v,'other')  for v in label]
    
    def create_tag_dict(self,dataset,rename_tag):
        """
        merge small class to 'other'
        """
        if rename_tag:
            map_=dataset.groupby('PRODUCT_TAG_ID')['PRODUCT_TAG_ID'].count().sort_values(ascending=False).to_frame('count')
            map_['label']=map_.index
            idx=map_['count']<10
            map_.loc[idx,'label']='other'
            dataset['label']=dataset['PRODUCT_TAG_ID'].map(map_['label'].to_dict())
        else:
            dataset['label']=dataset['PRODUCT_TAG_ID'].copy()
        return dataset.set_index('PRODUCT_TAG_ID')['label'].to_dict()


    def str_col_cat(self,data_his,cols_list):
        for k,col_name in enumerate(cols_list):
            cc=data_his[col_name].astype(str)
            if k==0:
                d_col=cc
            else:
                d_col=d_col.str.cat(cc,sep=' ')
        return d_col
    
    
    
#    @jit
    def filter_stop_and_stem_line(self,stemer,sentence,stop_words_dict):
#        line=' '.join([stemer.stem(w) for w in sentence.split(' ') if stop_words_dict.get(w) is  None ])
        line=' '.join([stemer.stem(w) for w in sentence.split(' ') ])
        return line
    
    def text_clean_batch(self,stemer,k,line_list, return_dict):
        '''worker function'''
        stop_words_dict={v:1 for v in stopwords.words('english')}
        stop_words_dict.update({'':1})
        stop_words_dict.pop('t')
        
        ret_list=[]
        for sentence in line_list:
            sentence=self.noise_clean_line(sentence)
            line=self.filter_stop_and_stem_line(stemer,sentence,stop_words_dict)
            ret_list.append(line)
    
        return_dict[k] = ret_list
    
        
   
#    @jit
    def noise_clean_line(self,line):
        """
        noise clean line
        """
        if not self.is_use_char:
            line=line.lower()
            line=re.sub("'s",'',line) ##delete 's
            line=re.sub('[0-9]{1,40}[.][0-9]*',' ^ ',line) ###match decimal
            line=re.sub('mp3','mp****',line)
            line=re.sub('mp4','mp****',line)
            line=re.sub('t[\s-]*shirt',' t**shirt',line)
            line=re.sub('powerbank',' power bank',line)
            line=re.sub('all[\s-]*in[\s-]*one','All**In**One',line)
    
            line=re.sub(',',';',line)
            line=re.sub('\.',';',line)
            line=re.sub(';',' ; ',line)
           
            add_punc = '.,;《》？！’ ” “”‘’@#￥% … &×（）——+【】{};；●，。°&～、|:：\n'
            punc = punctuation + add_punc
            punc=punc.replace('#','').replace('&','').replace('%','').replace('^','').replace('*','').replace(';','')
            line=re.sub(r"[{}]+".format(punc)," ",line)  ##delete dot
            line=re.sub('([2][0][0-9][0-9])','yyyy',line) ###match year 20**
            line=re.sub('\d{1,20}','*',line)  ##match int number
            
            line=re.sub('none','',line) ##deletel none
            line=re.sub('[\s]*[\s]',' ',line)  ##drop multi space
            line=re.sub('(^\s*)|(\s*$)','',line) ##delete head tail space
        else:
            line=line.lower()
            line=re.sub('[\s]*[\s]',' ',line) 
            line=' '.join([v for v in line])
        return line
    
    
    def to_batch_data(self,col,num):
        """
        col:list
        """
        batch_size=int(len(col)/num)
        batch_dict={}
        k=0
        if num>1:
            for k in range(num-1):
                line_list=col[k*batch_size:(k+1)*batch_size]
                batch_dict[k]=line_list
            line_list=col[(k+1)*batch_size:]
            batch_dict[k+1]=line_list
        else:
            line_list=col[k*batch_size:(k+1)*batch_size]
            batch_dict[k]=line_list
        return batch_dict
    
    
    def text_clean(self,col):
        """
        """
        logger.info('text_clean')
        t1=time.time()
        if type(col)!=list:
            col=col.tolist()
        
        
        if len(col)>10000:
            logger.info('use mutliprocess////')
            myprocess=multiProcess(col,worker=self.text_clean_batch)
            myprocess.set_tobatch(self.to_batch_data)
            col=myprocess.run()  
        else:
            logger.info('single process/////')
            num=1
            batch_dict=self.to_batch_data(col,num)
            return_dict={}
            stemer = PorterStemmer()
            for (k,line_list) in batch_dict.items():
                self.text_clean_batch(stemer,k,line_list,return_dict)
            ret_list=[]
            for k in range(num):
                ret_list.extend(return_dict[k])
            col=ret_list
        t2=time.time()
        logger.info('text_clean takes time///%s'%(t2-t1))
        return col
    
    def get_voc_dict(self,dataset):
        """
        product tag ==>voc dict
        """
        count_dict={}
        all_dict={}
        c=0
        for (key,td) in dataset.groupby('PRODUCT_TAG_ID'):
            c+=1
            voc_list=[]
            for v in td['text'].tolist():
                voc_list.extend(v.split(' '))
           
            vol_dict=dict(collections.Counter(voc_list))
            count_dict[key]=vol_dict
            all_dict[key]=voc_list
        return count_dict,all_dict
        
        
#    def create_data(self,data_his,all_dict):
#        """
#        create data 
#        """
#        add_list=[]
#        for key in all_dict:
#            word_list=list(all_dict[key])
#            sub=data_his[data_his['PRODUCT_TAG_ID']==key]
#            for v in range(len(sub)):
#                len_=np.random.randint(50,150) ###create text word size ,random
#                line=np.random.choice(word_list,len_).tolist()
#                line=[key, ' '.join(line)]
#                add_list.append(line)
#        add_pd=pd.DataFrame(add_list,columns=['PRODUCT_TAG_ID','text'])
#        return add_pd
        
#    def create_data(self,data_his,all_dict):
#        """
#        create data 
#        """
#        line_list=[]
#        for (text,tag) in data_his[['PRODUCT_NAME','PRODUCT_TAG_ID']].values:
#            line=text.split(' ')
#            rns.shuffle(line)
#            line=' '.join(line)
#            line_list.append([line,tag])
#        add_pd=pd.DataFrame(line_list,columns=['PRODUCT_TAG_ID','text'])
#        return add_pd
    
    
    def create_data(self,data_his,all_dict):
        """
        create data 
        """
        line_list=[]
        buy_data=data_his[data_his['BUYSELL']=='buy']
        sell_data=data_his[data_his['BUYSELL']=='sell']
        for tag_id in data_his['PRODUCT_TAG_ID'].unique():
            sub_buy =buy_data[buy_data['PRODUCT_TAG_ID']==tag_id]['PRODUCT_NAME'].tolist()
            sub_sell=sell_data[sell_data['PRODUCT_TAG_ID']==tag_id]['PRODUCT_NAME'].tolist()
            sub_buy=[''] if len(sub_buy)==0 else sub_buy
            sub_sell=[''] if len(sub_buy)==0 else sub_sell
            temp_list=[]
            if len(sub_buy)>1 and len(sub_sell)>1:
                for v in range(50):
                    new_line=[random.sample(sub_buy,1)[0],random.sample(sub_sell,1)[0]]
                    random.shuffle(new_line)
                    new_line=' '.join(new_line)
                    temp_list.append([new_line,tag_id])
                line_list.extend(temp_list)
        
#        for (text,tag) in data_his[['PRODUCT_NAME','PRODUCT_TAG_ID']].values:
#            line=text.split(' ')
#            rns.shuffle(line)
#            line=' '.join(line)
#            line_list.append([line,tag])
        add_pd=pd.DataFrame(line_list,columns=['PRODUCT_TAG_ID','text'])
        return add_pd
    
    def train_data_boost(self,data,is_add=True):
        logger.info('train_data_boost ,is_add:%s'%is_add)
        count_dict,all_dict=self.get_voc_dict(data)
        if is_add:
            add_pd=self.create_data(data,all_dict)
            add_pd['flag']='add'
            data_train=pd.concat([data[['PRODUCT_TAG_ID','text','flag']],add_pd],axis=0)
        else:
            data_train=data[['PRODUCT_TAG_ID','text','flag']]
        return data_train
      
        
    
    def save_obj(self,obj,obj_name):
        with open(os.path.join(self.data_model_dir,'%s.pickle'%obj_name), 'wb') as f:
            pickle.dump(obj, f)
            
    def read_obj(self,obj_name):
        with open(os.path.join(self.data_model_dir,'%s.pickle'%obj_name), 'rb') as f:
            return  pickle.load(f)            
 




           
#
#if __name__=='__main__':
#    model=TextCNN(seq_len=50,vocab_len=5000,voc_dim=150,num_classes=70,is_pre_train=False)
#    model.build_model()
#    print(22)


