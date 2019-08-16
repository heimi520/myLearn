#coding:utf8
import execjs
import requests
import logging
#import cx_Oracle
import json, time
from urllib.parse import urlencode
import pandas as pd
import re


COOKIE="NID=186=Qt78iteoiOwZDCIAK7F7BwF8JZjVsIE2ELGzbSLIxwZ1Cg_li7Z0S497gzec4yW67n9wcv2MCBobq3GC3EoS0CHx_0s0c3oYgsp4tG_vPdQSOmBDAmKXvqxbVPrbzlsaJRKMX8hZXOJ9nv5SDwz3SLUDxTKl_00DnTMUylwC20E; _ga=GA1.3.16817302.1561084149; 1P_JAR=2019-6-21-2"

'''翻译文本
=================

'''
logging.basicConfig(level = logging.INFO, format= '%(asctime)s %(levelname)s %(message)s', datefmt= '%Y-%m-%d %H:%M:%S')




def to_batch_data(pl_data):
    batch_line_list=[]
    batch_idx_list=[]
    batch_str_list=[]
    line_list=[]
    idx_list=[]
    str_cat=''
    for k,v in enumerate(pl_data['PRODUCT_NAME'].tolist()):
        str_cat+=(('; mytest%s@yeah.net; '%(k))+v)
        line_list.append(v)
        idx_list.append(k)
        if len(str_cat)>4300:
            batch_str_list.append(str_cat)
            batch_line_list.append(line_list)
            batch_idx_list.append(idx_list)
            ###reset batch
            str_cat=''
            line_list=[]  
            idx_list=[]
    ##append last batch
    batch_str_list.append(str_cat)
    batch_line_list.append(line_list) 
    batch_idx_list.append(idx_list)
    return batch_idx_list, batch_line_list,batch_str_list

   
    
def trans_batch(js,fromlang,tolang,batch_idx_list, batch_line_list,batch_str_list,batch_trans_list):
#    batch_trans_list=[]
    t0=time.time()
    for k,(idx_lines,lines,str_line) in enumerate(zip(batch_idx_list, batch_line_list,batch_str_list)):
        t1=time.time()
        transtext=js.translate(str_line,fromLang = fromlang, toLang = tolang)
        print('k////',k,'fromlang',fromlang,'tolang//',tolang,'trans len//',len(transtext))
#        break
        transtext2=re.sub(';','',transtext)  ##drop ;
        transtext2=re.sub('(^\s*)|(\s*$)','',transtext2)  ##drop head and tail space
        idx_pattern='mytest[0-9]{1,5}@yeah.net'
        transtext2=re.sub(idx_pattern,'!@#',transtext2) ##replace index flag to the same flag
    
        idx_list=[int(re.findall('[0-9]\d*',  v.replace(' ',''))[0]) for v in  re.findall(idx_pattern,transtext)] ##match idx
        trans_list=re.sub('(^\!@#)','',transtext2).split('!@#')                                          
        trans_pd=pd.DataFrame([idx_list,trans_list],index=['idx','trans_text']).T
        trans_pd['idx']=trans_pd['idx'].astype(int)
                                            
        line_pd=pd.DataFrame(lines,columns=['source_text'])
        line_pd['idx']=idx_lines
        
        merge_pd=pd.merge(line_pd,trans_pd,on=['idx'],how='left')
        merge_pd['batch_idx']=k    
        merge_pd['lang_from']=fromlang
        merge_pd['langto']=tolang                            
        batch_trans_list.append(merge_pd)
        t2=time.time()
        print('all batch',len(batch_line_list),'batch',k,'takes time',t2-t1,'total time',t2-t0)

#       
#    batch_trans_pd=pd.concat(batch_trans_list,axis=0)
#    batch_trans_pd['langfrom']=fromlang
#    batch_trans_pd['langto']=tolang
#    return batch_trans_pd






class MyGoogleTransTools(object):
    def __init__(self):
        self.jsctx=self.getJsctx()
        

    def  getJsctx(self):
        return execjs.compile(""" 
            function TL(a) { 
            var k = ""; 
            var b = 406644; 
            var b1 = 3293161072; 
    
            var jd = "."; 
            var $b = "+-a^+6"; 
            var Zb = "+-3^+b+-f"; 
    
            for (var e = [], f = 0, g = 0; g < a.length; g++) { 
                var m = a.charCodeAt(g); 
                128 > m ? e[f++] = m : (2048 > m ? e[f++] = m >> 6 | 192 : (55296 == (m & 64512) && g + 1 < a.length && 56320 == (a.charCodeAt(g + 1) & 64512) ? (m = 65536 + ((m & 1023) << 10) + (a.charCodeAt(++g) & 1023), 
                e[f++] = m >> 18 | 240, 
                e[f++] = m >> 12 & 63 | 128) : e[f++] = m >> 12 | 224, 
                e[f++] = m >> 6 & 63 | 128), 
                e[f++] = m & 63 | 128) 
            } 
            a = b; 
            for (f = 0; f < e.length; f++) a += e[f], 
            a = RL(a, $b); 
            a = RL(a, Zb); 
            a ^= b1 || 0; 
            0 > a && (a = (a & 2147483647) + 2147483648); 
            a %= 1E6; 
            return a.toString() + jd + (a ^ b) 
        }; 
    
        function RL(a, b) { 
            var t = "a"; 
            var Yb = "+"; 
            for (var c = 0; c < b.length - 2; c += 3) { 
                var d = b.charAt(c + 2), 
                d = d >= t ? d.charCodeAt(0) - 87 : Number(d), 
                d = b.charAt(c + 1) == Yb ? a >>> d: a << d; 
                a = b.charAt(c) == Yb ? a + d & 4294967295 : a ^ d 
            } 
            return a 
        } 
        """)
        
    def translate(self,content, tk = None, fromLang = 'auto', toLang = 'en', proxies = None):
        '''用GOOGLE翻译
        ================
        当源字符有换行（\n）时，返回一个也会有换行
        '''
        time.sleep(0.5)
        res=''
        if len(content) > 4891:
            print(u"翻译的长度超过限制！！！")
            return res
        if tk is None:
            tk = self.jsctx.call("TL", content)
        param = {'tk': tk, 'q': content}  
        headers = {
            'accept': '*/*'
            , 'accept-encoding': 'gzip, deflate, br'
            , 'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8'
            , 'cache-control': 'no-cache'
            , 'cookie': COOKIE
            , 'pragma': 'no-cache'
            , 'referer': 'https://translate.google.cn/?source=osdd'
            , 'user-agent': 'Mozilla/5.0 (Linux; Android 4.0.4; Galaxy Nexus Build/IMM76B) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.133 Mobile Safari/535.19'
        }
        param = dict(param, **{
            'client': 'webapp'
            , 'sl': fromLang
            , 'tl': toLang
            , 'hl': toLang
            , 'dt': 'at'
            , 'dt': 'bd'
            , 'dt': 'ex'
            , 'dt': 'ld'
            , 'dt': 'md'
            , 'dt': 'qca'
            , 'dt': 'rw'
            , 'dt': 'rm'
            , 'dt': 'ss'
            , 'dt': 't'
            , 'otf': '1'
            , 'ssel': '0'
            , 'tsel': '3'
            , 'kc': '6'
        })
        c=0
        while True:
            c+=1
            if c>1:
                print('retry times///',c)
                time.sleep(1)
            if c>5:
                break
            try:
                r = requests.get("""https://translate.google.cn/translate_a/single""", \
                                 headers = headers,\
                                 params=param,\
                                 timeout = 10,\
                                 proxies = proxies)
                result = r.json()
                if r.status_code == 200:
                    res= ''.join([t[0] for t in result[0] if t[0] is not None])
                    break
            except Exception as e:
                print('error///',e)
        return res
#






#
#
#js=MyGoogleTransTools()
#
#import pandas as pd
#
#small_data=pd.read_excel('../data/lang_trans/tag .xlsx')
#
#line_list=list(set(small_data['T1'].tolist()+small_data['T2'].tolist()+small_data['Product Tag'].tolist()))
#line_se=pd.Series(line_list)
#subdata=pd.DataFrame(line_se[line_se.notnull()],columns=['PRODUCT_NAME'])
#
###########################################################################
#
#subdata.index=range(len(subdata))
#
#ret_list=[]
#for k, str_line in enumerate(subdata['PRODUCT_NAME']):
#    print(k)
#    transtext=js.translate(str_line,fromLang ='en', toLang = 'ar')
#    ret_list.append([str_line,transtext])
#    
#
#ret_pd=pd.DataFrame(ret_list,columns=['source','trans'])
#
#
