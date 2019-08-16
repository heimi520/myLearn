# -*- coding: utf-8 -*-



from scrapy import signals
import time,random,logging
import  requests
from datetime import datetime
import random

'''
这个类主要用于产生随机UserAgent
'''
class RandomUserAgent(object):

    def __init__(self,agents):
        self.agents = agents

    @classmethod
    def from_crawler(cls,crawler):
        return cls(crawler.settings.getlist('USER_AGENTS')) #返回的是本类的实例cls ==RandomUserAgent

    def process_request(self,request,spider):
        request.headers.setdefault('User-Agent', random.choice(self.agents))


class RandomDelayMiddleware(object):
    def __init__(self, delay):
        self.delay = delay

    @classmethod
    def from_crawler(cls, crawler):
        delay = crawler.spider.settings.get("RANDOM_DELAY", 10)
        if not isinstance(delay, int):
            raise ValueError("RANDOM_DELAY need a int")
        return cls(delay)

    def process_request(self, request, spider):
        delay = random.randint(0, self.delay)*0.2
        logging.debug("### random delay: %s s ###" % delay)
        time.sleep(delay)



class Proxy(object):
    '''网络代理
    ============
    '''
    def __init__(self, useProxy, useSockt5=False):
        self.lastProxy = None
        self.lastVistTime = None
        self.useProxy = useProxy
        self.useSockt5 = useSockt5

    def getProxy(self, revisitTime=60):
        '''获取代理地址
        ===============
        返回代理服务器IP，端口号

        参数
        ------
        **revisitTime** 超过这个秒数才重新访问
        '''
        if self.useSockt5:
            return {"http": "socks5://127.0.0.1:9150", "https": "socks5://127.0.0.1:9150"}
        if not self.useProxy: return None
        # url = r'http://proxy.httpdaili.com/apinew.asp?text=true&noinfo=true&sl=1&ddbh=lan88'
        url = r'http://proxy.httpdaili.com/apinew.asp?text=true&noinfo=true&sl=1&ddbh=lan991'
        if self.lastVistTime is None:
            self.lastVistTime = datetime.now()
        else:
            if (
                    datetime.now() - self.lastVistTime).seconds < revisitTime and self.lastProxy is not None: return self.lastProxy
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




class ProxyMiddleware(object):
    '''
    设置Proxy
    '''
    def __init__(self,ip_list):
        self.proxies = Proxy(True, False)
        self.ip=''
        self.t0=time.time()

    @classmethod
    def from_crawler(cls, crawler):
        return cls(ip_list=crawler.settings.get('PROXIES'))

    def process_request(self, request, spider):
        try:
            # ip = random.choice(self.ip_list)
            ip=self.proxies.getProxy(5)['http']
            if ip!=self.ip:
                self.ip=ip
                t2=time.time()
                print('change ip//////////////////////////',ip,'takes time',t2-self.t0)
            request.meta['proxy'] = self.ip

        except  Exception as e:
            print('error!!!!!!!!!!!!!!!!',e)



