# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:28:05 2019

@author: Administrator
"""



import warnings
warnings.filterwarnings('ignore')

from model_meorient import *
import pandas as pd


class DetectText2Feature(Text2Feature):
    cols_list=['PRODUCT_NAME']
    
    def __init__(self,seq_len,max_words=None,is_rename_tag=False,is_add_data=False,model_id=None):
        Text2Feature.__init__(self,seq_len=seq_len,max_words=max_words, is_rename_tag=is_rename_tag,is_add_data=is_add_data,model_id=model_id)
        
    def text_clean(self,col):
        """
        """
        logger.info('text_clean')
        col=col.str.lower()
        col=col.str.replace('[0-9]','') ###delete num
#        
        add_punc = '.,;《》？！’ ” “”‘’@#￥% … &×（）——+【】{};；●，。°&～、|:：\n'
        punc = punctuation + add_punc
        col=col.apply(lambda x: re.sub(r"[{}]+".format(punc)," ",x))  ##delete dot

        col=col.str.replace('(^\s*)|(\s*$)','') ##delete head tail space
        col=col.str.replace(' ','-')
        col=col.apply(lambda x:' '.join([v for v in x]))
#
        return col    
   
    





class TextCNN(object):
    model_net_filename='textcnn_net'
    model_w_filename='textcnn_w'
    
    def __init__(self,seq_len=None,max_words=None,\
                 voc_dim=None,num_classes1=None,num_classes2=None, tokenizer=None, cut_list=[],\
                 is_pre_train=False, init_learn_rate=0.001,batch_size=512,\
                 epoch_max=50,drop_out_rate=0.5,early_stop_count=5,\
                 model_id='base_model'):
        
        self.kernel_sizes =[1,2,2]##[1,2,2,3]##[2,3,4]##[1,1,2,2,3,4]   ##[1,2,3,4,5,6] ##[2,3,4,5,6,7] ##[2,3,4,6,8] ##[2,3,4,5,6] ### [2,2,3,3,4,4]
        self.filter_num=64
        self.max_words=max_words
        self.seq_len=seq_len
        self.voc_dim=voc_dim
        self.is_pre_train=is_pre_train
        self.num_classes1=num_classes1
        self.num_classes2=num_classes2
        self.tokenizer=tokenizer
        self.cut_list=cut_list
        self.drop_out_rate=drop_out_rate/2 if is_pre_train  else drop_out_rate
        self.init_learn_rate=init_learn_rate
        self.early_stop_count=early_stop_count
        
        self.batch_size=batch_size
        self.epoch_max=epoch_max
        self.model=None  ###model init
        
        self.data_model_dir=os.path.join('../data/temp' ,model_id) 
        if not os.path.isdir(self.data_model_dir):
            os.makedirs(self.data_model_dir)  
        
        logger.info('TextCNN  __init__ params: max_words:%s'%max_words)
        logger.info('TextCNN  __init__ params: seq_len:%s'%seq_len)
        logger.info('TextCNN  __init__ params: voc_dim:%s'%voc_dim)
        logger.info('TextCNN  __init__ params: num_classes1:%s'%num_classes1)
        logger.info('TextCNN  __init__ params: num_classes2:%s'%num_classes2)
        logger.info('TextCNN  __init__ params: cut_list:%s'%cut_list)
        logger.info('TextCNN  __init__ params: is_pre_train:%s'%is_pre_train)
        logger.info('TextCNN  __init__ params: init_learn_rate:%s'%init_learn_rate)
        logger.info('TextCNN  __init__ params: batch_size:%s'%batch_size)
        logger.info('TextCNN  __init__ params: epoch_max:%s'%epoch_max)
        logger.info('TextCNN  __init__ params: drop_out_rate:%s'%self.drop_out_rate)
        logger.info('TextCNN  __init__ params: early_stop_count:%s'%early_stop_count)
        logger.info('TextCNN  __init__ params: model_id:%s'%model_id)
#        logger.info('TextCNN  __init__ params: max_words:%s'%)
        
#        
#        logger.info( ("TextCNN  __init__ params:,max_words:%s,seq_len=%s,vocab_len=%s, voc_dim=%s, num_classes=%s, is_pre_train=%s,\
#                 init_learn_rate=%s,batch_size=%s,epoch_max=%s,drop_out_rate=%s, early_stop_count=%s,model_id=%s"%(\
#                 max_words,seq_len,vocab_len,voc_dim,num_classes,is_pre_train,init_learn_rate,batch_size,epoch_max,\
#                 drop_out_rate,early_stop_count,model_id)).replace(' ','').replace(',','\n') )
        
#    


    def Tag2T1(self,x,cut_list):
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
            max_=Reshape([1])(K.max(line,axis=1))
            ret_list.append(max_)
        out=keras.layers.concatenate(ret_list, axis=1)
        return out    
    
    
    def load_embed(self,tokenizer):
        model=word2vec.Word2Vec.load(os.path.join(self.data_model_dir,'word2vec.model'))
        line_list=[]
        for k, v in  enumerate(sorted(tokenizer.word_index ,key=lambda x:[1],reverse=True)[:self.max_words+1]):
            try:
                line=model[v]
            except:
                line=np.random.random(model.vector_size)
            line_list.append(line)
        embed_w=np.array(line_list)
        np.savetxt(os.path.join(self.data_model_dir,'embed.csv'),embed_w)
        return embed_w


    def build_model(self):
        """
        """
        ##textcnn####conv1
        place_x= Input(shape=(self.seq_len,),name='x_input')
        if self.is_pre_train:
            embed_w=self.load_embed(self.tokenizer)
        #    embed_w=model.get_layer('embedding_1').get_weights()[0]
        #    np.savetxt('../data/input/embed.csv',embed_w)
#            embed_w=np.loadtxt(os.path.join(self.data_model_dir,'embed.csv'),'float32')
        embed = keras.layers.Embedding(self.max_words+1, ###self.vocab_len + 1,
                                       self.voc_dim ,
                                       trainable= True if not self.is_pre_train  else False,
                                       weights=[embed_w] if self.is_pre_train else None , 
                                       input_length=self.seq_len)(place_x)
        
        shape_=(embed.shape[1].value,embed.shape[2].value)
        embed_reverse= Lambda(lambda x:x[:,::-1,:],output_shape=shape_)(embed)

        lay1_list=self.create_conv_pool(embed,self.kernel_sizes,self.filter_num,self.seq_len)
        lay2_list=self.create_conv_pool(embed_reverse,self.kernel_sizes,self.filter_num,self.seq_len)
        lay_list=lay1_list+lay2_list
        net = keras.layers.concatenate(lay_list, axis=1)
        
        net =Dropout(self.drop_out_rate)(net)
        net= Dense(256)(net)
       
        net=BatchNormalization(axis=1)(net)
        net=LeakyReLU(alpha=0.05)(net)
        
        net =Dropout(self.drop_out_rate)(net)
        out2=Dense(self.num_classes2)(net) ###TAG
        output2=Activation('softmax',name='output2')(out2)
        
        
        out1_1 = Lambda(self.Tag2T1,arguments={'cut_list':self.cut_list} )(out2) ####T1
        output1_1=Activation('softmax',name='output1_1')(out1_1)
        
#        out1_2=Dense(self.num_classes1)(net) ###TAG
#        output1_2=Activation('softmax',name='output1_2')(out1_2)
#    

        self.model = keras.models.Model(inputs=place_x, outputs=[output1_1,output2])
        self.model.summary()
        
        opt=keras.optimizers.Adam(lr=self.init_learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08,clipvalue=1.0)
        self.model.compile(loss={'output1_1': 'categorical_crossentropy', \
                                 'output2': 'categorical_crossentropy'},\
                                   optimizer=opt,\
                                   metrics=['accuracy'],\
                                   loss_weights={'output1_1': 1, 'output2': 1} \
                                   )

 

        return self.model

        
    def train(self,x_train_padded_seqs, y1_train,y2_train, x_val_padded_seqs, y1_val,y2_val,w_sp_train=None):
        """
        """
        logger.info('training model....')
        self.model.save(os.path.join(self.data_model_dir,'%s.h5'%self.model_net_filename))
        #记录所有训练过程，每隔一定步数记录最大值
        tensorboard = TensorBoard(log_dir=self.data_model_dir)
        checkpoint = ModelCheckpoint(os.path.join(self.data_model_dir,'%s.h5'%self.model_w_filename),
                                     monitor="val_output2_acc",
                                     mode='max',
                                     save_weights_only=True,
                                     save_best_only=True, 
                                     verbose=0,###no logs
                                     period=1)
         
                 
        monitor=TrainingMonitor()
        earlystop=EarlyStopping(monitor='val_output2_acc', patience=self.early_stop_count,mode='max', verbose=2)
        callback_lists=[tensorboard,checkpoint,earlystop,monitor]
        ##########################################################
        self.history=self.model.fit(
                                    x={'x_input':x_train_padded_seqs}, 
                                    y={'output1_1':y1_train,'output2':y2_train}, 
                                    batch_size=self.batch_size,              
                                    initial_epoch=0,
                                    epochs=self.epoch_max, 
                                    validation_data = (x_val_padded_seqs, [y1_val, y2_val]),
                                    verbose=0,
                                    callbacks=callback_lists,
                                    sample_weight=[w_sp_train,w_sp_train],##3 output
                                    shuffle=True,
                                    
#                                        workers=4,
#                                        use_multiprocessing=True,
                                    )
        ##############################################################################
#        self.history=self.model.fit_generator(
#                                        self.generate_batch_data_random(x_train_padded_seqs, y1_train,y2_train, self.batch_size),                                                      
#                                        steps_per_epoch=len(x_train_padded_seqs)//self.batch_size,
#                                        initial_epoch=0,
#                                        nb_epoch=self.epoch_max, 
#                                        validation_data = (x_val_padded_seqs, y1_val,y2_val),
#                                        verbose=0,
#                                        callbacks=callback_lists,
##                                        sample_weight=w_sp_train,
#                                        
##                                        workers=4,
##                                        use_multiprocessing=True,
#                                        )
        return self.history

#
#    def partical_predict(self,x_test_padded_seqs):
#        self.model=load_model(os.path.join(self.data_model_dir,'%s.h5'%self.model_net_filename))   # H    
#        self.model.load_weights(os.path.join(self.data_model_dir,'%s.h5'%self.model_w_filename))
#        
#        return self.model.predict(x_test_padded_seqs) 
#    
            
    def predict(self,x_test_padded_seqs):
        if self.model is None:
            self.model=load_model(os.path.join(self.data_model_dir,'%s.h5'%self.model_net_filename))   # H    
        self.model.load_weights(os.path.join(self.data_model_dir,'%s.h5'%self.model_w_filename))
        return self.model.predict({'x_input':x_test_padded_seqs},batch_size=self.batch_size) 
    
    def predict_classes(self,x_test_padded_seqs):
        prob_list=self.predict(x_test_padded_seqs)
        pred_int_list=[]
        for prob in  prob_list:
            y_pred_int=np.argmax(prob,axis=1)
            pred_int_list.append(y_pred_int)
        return pred_int_list 
    
    
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
                                       strides=1,padding='valid',\
                                       use_bias=True)(embed)  
            
            conv=BatchNormalization(axis=2)(conv)
            conv=LeakyReLU(alpha=0.05)(conv)
            
            pool = keras.layers.MaxPooling1D(pool_size=seq_len-ks+1,strides=1,padding='valid')(conv) 
            pool = keras.layers.Flatten()(pool)
            lay_list.append(pool)
        return lay_list


    
        
    
    def generate_batch_data_random(self,x, y1,y2, batch_size):
        """逐步提取batch数据到显存，降低对显存的占用"""
        batches = (len(y1) + batch_size - 1)//batch_size
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
                Y1 = y1[i*batch_size : (i+1)*batch_size]
                Y2 = y2[i*batch_size : (i+1)*batch_size]
                
#                rns.shuffle(idx_col)
#                X=X[:,idx_col]
                
                yield [X], [Y1,Y2]
    




