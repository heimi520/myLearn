# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:43:38 2019

@author: Administrator
"""



import requests
import json
from bs4 import BeautifulSoup
import execjs #必须，需要先用pip 安装，用来执行js脚本
import time

class Py4Js(object):   
    """
    https://www.jianshu.com/p/95cf6e73d6ee
    """
    def __init__(self):  
        self.ctx = execjs.compile(""" 
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
    def getTk(self,text):  
        return self.ctx.call("TL",text)
    
    def buildUrl(self,text,tk):
        baseUrl='https://translate.google.cn/translate_a/single'
        baseUrl+='?client=t&'
        baseUrl+='sl=auto&'
        baseUrl+='tl=en&'
        baseUrl+='hl=auto&'
        baseUrl+='dt=at&'
        baseUrl+='dt=bd&'
        baseUrl+='dt=ex&'
        baseUrl+='dt=ld&'
        baseUrl+='dt=md&'
        baseUrl+='dt=qca&'
        baseUrl+='dt=rw&'
        baseUrl+='dt=rm&'
        baseUrl+='dt=ss&'
        baseUrl+='dt=t&'
        baseUrl+='ie=UTF-8&'
        baseUrl+='oe=UTF-8&'
        baseUrl+='otf=1&'
        baseUrl+='pc=1&'
        baseUrl+='ssel=0&'
        baseUrl+='tsel=0&'
        baseUrl+='kc=2&'
        baseUrl+='tk='+str(tk)+'&'
        baseUrl+='q='+text
        return baseUrl
    
    def translate(self,text):
        header={
                'authority':'translate.google.cn',
                'method':'GET',
                'path':'',
                'scheme':'https',
                'accept':'*/*',
                'accept-encoding':'gzip, deflate, br',
                'accept-language':'zh-CN,zh;q=0.9',
                'cookie':'',
                'user-agent':'Mozilla/5.0 (Windows NT 10.0; WOW64)  AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.108 Safari/537.36',
            'x-client-data':'CIa2yQEIpbbJAQjBtskBCPqcygEIqZ3KAQioo8oBGJGjygE='
              }
        url=self.buildUrl(text,self.getTk(text))
        
        res=''
        c=0
        while True:
            c+=1 
            if c>1:
                print('count///',c)
                time.sleep(1.0)
            if c>5:
                break
            try:
                r=requests.get(url)
                result=json.loads(r.text)
                if result[7]!=None:
                      # 如果我们文本输错，提示你是不是要找xxx的话，那么重新把xxx正确的翻译之后返回
                    try:
                        correctText=result[7][0].replace('<b><i>',' ').replace('</i></b>','')
                        print(correctText)
                        correctUrl=buildUrl(correctText,js.getTk(correctText))
                        correctR=requests.get(correctUrl)
                        newResult=json.loads(correctR.text)
                        res=newResult[0][0][0]
                        break
                    except Exception as e:
                        print('error1///',e)
                        res=result[0][0][0]
                    
                else:
                    res=result[0][0][0]
                    break
                        
            except Exception as e:
                res=''
                print(url)
                print("翻译"+text+"失败")
                print("错误信息:")
                print('error2//',e)

        return res
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
##if __name__ == '__main__':
#js=Py4Js()
#for v in range(1000):
# 
#    res=js.translate('Они граждане.')
#    print('line//',v,res)
#    time.sleep(0.5)
#    
    
    
    
#a=[[['They are Georgian citizens, these are my mothers, and from the Russian leadership have exactly the same trouble as all other Georgian citizens.', 'Они грузинские гражданки, эти мои мамы, и от российского руководства имеют ровно столько же неприятностей, сколько все остальные грузинские граждане.', None, None, 3], [None, None, None, "Oni gruzinskiye grazhdanki, eti moi mamy, i ot rossiyskogo rukovodstva imeyut rovno stol'ko zhe nepriyatnostey, skol'ko vse ostal'nyye gruzinskiye grazhdane."]], None, 'ru', None, None, [['Они грузинские гражданки, эти мои мамы, и от российского руководства имеют ровно столько же неприятностей, сколько все остальные грузинские граждане.', None, [['They are Georgian citizens, these are my mothers, and from the Russian leadership have exactly the same trouble as all other Georgian citizens.', 0, True, False], ['They Georgian citizen, these are my mother, and from the Russian leadership have exactly the same amount of trouble as all other Georgian citizens.', 0, True, False]], [[0, 149]], 'Они грузинские гражданки, эти мои мамы, и от российского руководства имеют ровно столько же неприятностей, сколько все остальные грузинские граждане.', 0, 0]], 1, None, [['ru'], None, [1], ['ru']]]


    