#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 09:03:21 2019

@author: heimi
"""


class Proxy(object):
    '''网络代理
    ============
    '''
    def __init__(self, useProxy, useSockt5 = False):
        self.lastProxy = None
        self.lastVistTime = None
        self.useProxy = useProxy
        self.useSockt5 = useSockt5
        
    def getProxy(self, revisitTime = 60):
        '''获取代理地址
        ===============
        返回代理服务器IP，端口号
        
        参数
        ------
        **revisitTime** 超过这个秒数才重新访问
        '''
        if self.useSockt5:
            return {"http":"socks5://127.0.0.1:9150","https":"socks5://127.0.0.1:9150"}
        if not self.useProxy: return None
        #url = r'http://proxy.httpdaili.com/apinew.asp?text=true&noinfo=true&sl=1&ddbh=lan88'
        url = r'http://proxy.httpdaili.com/apinew.asp?text=true&noinfo=true&sl=1&ddbh=lan991'
        if self.lastVistTime is None:
            self.lastVistTime = datetime.now()
        else:
            if (datetime.now() - self.lastVistTime).seconds < revisitTime and self.lastProxy is not None: return self.lastProxy
        try:
            r = requests.get(url)
            if r.status_code == 200:
                proxies = {
                    'http': 'http://%s' % r.text.strip(), 
                    'https': 'https://%s' % r.text.strip() 
                        }
                self.lastProxy = proxies
                return self.lastProxy
        except ConnectionError as e:
            return self.lastProxy
        return self.lastProxy
    
from datetime import datetime
proxies=Proxy(True, False)    
aa= proxies.getProxy(12)['http'] if proxies is not None and proxies.getProxy(12) is not None else None

print(aa)
    


