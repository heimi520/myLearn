# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:34:51 2019

@author: Administrator
"""



from model_meorient import *
from model_config import *


#
#class TextDemo(Text2Feature):
#    def __init__(self,seq_len,token_path,is_rename_tag=True,is_add_data=False):
#        Text2Feature.__init__(self,seq_len,token_path,is_rename_tag=True,is_add_data=False)
#        
#        
#    def test(self,data_train,data_val):
#        data_train['flag']='train'
#        data_val['flag']='val'
#        data=pd.concat([data_train,data_val],axis=0)   
## 
#        data['text']=self.__str_col_cat(data,self.cols_list)
#        
##        self.dd= self.__text_clean(data['text'])
##        
##        
#        
#



import pandas as pd
from sklearn.model_selection import train_test_split

print('data text clean....')
data_his=pd.read_csv('../data/input/data_his.csv')
data_train,data_val= train_test_split(data_his,test_size=0.1)


COUNTRY_LIST=['Mexico',
             'Turkey',
             'Nigeria',
             'Poland',
             'Kazakhstan',
             'United Arab Emirates',
             'Kenya',
             'South Africa',
             'Brazil',
             'Switzerland',
             'India',
             'Egypt']    
    
COUNTRY_LANGUAGE_DICT={'Mexico':'spanish',
 'Turkey':'turkish',
  'Nigeria':'english',
  'Poland':'polish',
  'Kazakhstan':'kazakh',
 'United Arab Emirates':'arabic',
 'Kenya':'english',
 'South Africa':'english',
 'Brazil':'english',
 'Switzerland':'english',
  'India':'english',
 'Egypt':'arabic'
 }    
 

#country_pd=pd.DataFrame(COUNTRY_LIST,columns=['COUNTRY_NAME'])
#data_his=pd.merge(data_his,country_pd,on=['COUNTRY_NAME'],how='inner')



col=data_his['PRODUCT_NAME']

import hashlib
import random
import requests
import time
 


def translate(source):
    myurl = 'http://api.fanyi.baidu.com/api/trans/vip/translate'
    salt = random.randint(32768, 65536)
    appid = '20190610000306216' #你的appid
    secretKey ='kpiKJNQM9D_28RLDoNfH' #你的密钥
    
    def create_md5(appid,secretKey,salt,source):
        sign = appid+source+str(salt)+secretKey
        m1 = hashlib.md5()
        m1.update(sign.encode('utf-8'))
        sign = m1.hexdigest()
        return sign
    
    
    params={
            'q' : source.encode('utf-8'),
            'from' : 'auto',
            'to' : 'en',
            'salt' : salt,
            'sign' : create_md5(appid,secretKey,salt,source),
            'appid' : '20190610000306216', #你的appid
            'secretKey' : 'kpiKJNQM9D_28RLDoNfH', #你的密钥
            }
    
    t1=time.time()
    res=requests.post(myurl,data=params,timeout=5)
    t2=time.time()
    print('takes time',t2-t1)
    text=res.json()['trans_result'][0]['dst']
    return text
        
        
def text_clean(col):
    """
    """
    col=col.str.replace('([2][0][0-9][0-9])','****') ###match year 20**
    col=col.str.replace('[0-9]{1,40}[.][0-9]*','^') ###match decimal
    col=col.str.replace('\d{1,20}','*')  ##match int number
    col=col.str.replace("'s",'').str.lower() ##delete 's
    col=col.str.replace('none','') ##deletel none
    
    add_punc = '.,;《》？！’ ” “”‘’@#￥% … &×（）——+【】{};；●，。&～、|\s:：'
    punc = punctuation + add_punc
    punc=punc.replace('*','').replace('^','') ##dot set
    
    col=col.apply(lambda x: re.sub(r"[{}]+".format(punc)," ",x))  ##delete dot
    col=col.str.replace('(^\s*)|(\s*$)','') ##delte head and tail space
        

    return col
    
col=data_his['PRODUCT_NAME'].astype(str)
col2=text_clean(col)
data_his['text']=col2
aa=data_his[['PRODUCT_NAME','text']]


stemer = PorterStemmer()

voc_dict={}
for v in data_his.groupby('LANGUAGE'):
    laguage=v[0]
    td=v[1]
    line_list=[]
    for vv in td['text'].values:
        line_list.extend(vv.split(' '))
    voc_dict[laguage]=line_list



voc_set_dict={}
other_dict={}
for language in voc_dict:
    cc=list(set(voc_dict[language]))
    cc_pd=pd.DataFrame(cc,columns=['word'])
    break
    add_punc = '.,;《》？！’ ” “”‘’@#￥% … &×（）——+【】{};；●，。&～、|\s:：'
    punc = punctuation + add_punc
    
    
    cc_pd['word2']=cc_pd['word'].apply(lambda x: re.sub(r"[{}]+".format(punc)," ",x)) 
    cc_pd['word2']=cc_pd['word2'].apply(lambda x: re.sub(r'[\a-zA-Z]'," ",x)) 
    cc_pd['word2']=cc_pd['word2'].str.replace('(^\s*)|(\s*$)','').str.replace(' ','')
    
    
    other=cc_pd[cc_pd['word2'].apply(lambda x:len(str(x)) )>0]
    text_list=[]
    for source in other['word']:
        break
        source=','.join(other['word'].tolist())
        translate('good ,good, study ,day, day, up')
        line=translate(source)
#        lang=translator.detect(source).lang
        text = translator.translate(source,src=language,dest='en').text
        text_list.append(lang)
        time.sleep(0.5)
        print('languge:%s source:%s text:%s '%(language,source,lang))
    
    other['text_dest']=text_list
#    break
    other_dict[language]=other
    
    
    
    
    cc_pd['stem']=cc_pd['word'].apply(lambda x:stemer.stem(x))
    voc_set_dict[language]=cc_pd
#    voc_set_dict[k]=dict(collections.Counter(voc_dict[k]))

#
#
#demo=TextDemo(seq_len=SEQ_LEN,token_path=TOKEN_PATH,is_rename_tag=True,is_add_data=False)
#demo.test(data_train,data_val)




