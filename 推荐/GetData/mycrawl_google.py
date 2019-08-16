#coding:utf8
import asyncio, aiohttp
from lxml import etree
import logging, json
import requests
from sqlalchemy import create_engine
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
import pandas as pd
import codecs, random
from datetime import datetime
from aiohttp_socks import SocksConnector

'''从https://www.amazon.com/?language=en_US找关键字的产品名称
===============================================================
created by wlq
created on 2019-6-14
'''
#页面编码
CODINGING = 'utf8'

#2014年1月21日mongo db 服务器IP更改，原来是192.168.1.233
DB_HOST = '10.20.8.113'
DB_NAME = 'expdata'
DB_PORT = 27017
BATCH_ID = '20190614'
#DB_COLLECTION = 'amazon_com_products'
#CONTACT_FILE = '%s %s.xlsx' % (DB_COLLECTION, BATCH_ID)
logging.basicConfig(level = logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

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

def getXpath(tree, xpath, firstOne = False):
    '''获取一个XPATH内容
    ====================
    '''
    eles = tree.xpath(xpath)
    if eles is not None and firstOne and len(eles): return eles[0]
    return eles

def getEleTxt(ele, replaceStr = ':：'):
    '''迭代找出元素所包含的text
    ===========================
    
    参数
    -----
    **ele** 要迭代的根元素
    **** 要替换掉的字符 unicode类型
    '''
    if isinstance(ele, str): 
        return ele
    else:
        txt = u' '.join(ele.itertext()).replace(u'\n', u'').replace(u'\r', u'').replace(u'  ', u'')
        for s in replaceStr:
            txt = txt.replace(s, u'')
        return txt
    
def getRequest(url, headers, method = 'get', data = None, proxies = None, timeout = 120, returnHTML = False, retrySeconds = 1):
    '''请求一个url返回xlm树
    =====================================
    '''
    proxies = proxies.getProxy() if proxies is not None else None
    while True:
        try: 
            if method == 'get':
                r = requests.get(url, headers = headers, params= data, proxies = proxies, timeout = timeout)
            elif method == 'post':
                r = requests.post(url, headers = headers, data= data, proxies = proxies, timeout = timeout)
            if r.status_code == 200:
                if returnHTML: return r, True
                return etree.HTML(r.text), True
            else:
                logging.error('ERR')
                return r.status_code, False
        except Exception as e:
            logging.error(e)
            time.sleep(retrySeconds)
    return r.status_code, False

async def asyncRequest(url, semaphore, session = None, method = 'get', data = None, headers = None, proxies = None, timeout = 120, returnHTML = False, retrySeconds = 1, callback = None):
    '''异步请求
    =============
    
    参数
    ----------
    **url** 访问的地址
    **semaphore** 并发控制
    **session** async session, 可以为None
    **method** get, post 不区分大小写
    **data** get的params参数， 或post的data参数
    **headers** headers
    **proxies** Proxy类的实例
    **timeout** 超时秒数
    **returnHTML** 给callback函数调用的参数是html源内容，还是XML tree对象，默认为XML tree对象
    **retrySeconds** 出错重试前暂停的秒数
    **callback** 请求完成后执行的函数, 是一个协程函数
    '''
    async with semaphore:
        while True:
            proxy = proxies.getProxy(12)['http'] if proxies is not None and proxies.getProxy(12) is not None else None
            try:
                if session is None:
                    async with aiohttp.request(method.lower(), url, headers = headers, params = data, proxy = proxy) as response:
                        if 200 == response.status:
                            html = await response.text()
                            html = '<html><head></head><body>%s</body></html>' % html
                            if returnHTML: 
                                await callback(html, data.get('k', ''), data.get('page', ''))
                                break
                            await callback(etree.HTML(html), data.get('k', ''), data.get('page', ''))
                            break
                else:
                    if method.lower() == 'get':
                        async with session.get(url, headers = headers, params = data, timeout = timeout, proxy = proxy) as response:
                            if 200 == response.status:
                                html = await response.text()
                                html = '<html><head></head><body>%s</body></html>' % html
                                if returnHTML: 
                                    await callback(html, data.get('k', ''), data.get('page', ''))
                                    break
                                await callback(etree.HTML(html), data.get('k', ''), data.get('page', ''))
                                break
                    elif method.lower() == 'post':
                        async with session.post(url, headers = headers, data = data, timeout = timeout, proxy = proxy) as response:
                            if 200 == response.status:
                                html = await response.text()
                                if returnHTML: 
                                    await callback(html, data.get('k', ''), data.get('page', ''))
                                    break
                                await callback(etree.HTML(html), data.get('k', ''), data.get('page', ''))
                                break
                break
            except Exception as e:
                logging.error('出错(停几秒后重试)%s %s' % (data, e))
                if proxies is not None and proxy is not None:
                    while proxy == proxies.getProxy(12)['http']:
                        await asyncio.sleep(1)
                else:
                    await asyncio.sleep(random.randint(1, retrySeconds))
    
async def fetchdata(multi = 5):
    '''数据摘录
    ==============

    '''
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3'
        , 'Accept-Encoding': 'gzip, deflate'
        , 'Accept-Language': 'en-US,us;q=0.9,en;q=0.8'
        , 'Cache-Control': 'no-cache'
        #, 'Cookie': 's_fid=04967EB421E59264-081556DBCAAFB5A4; regStatus=pre-register; c_m=undefinedwww.google.comSearch%20Engine; s_cc=true; aws-priv=eyJ2IjoxLCJldSI6MCwic3QiOjB9; aws_lang=cn; session-id=135-8983574-0716440; session-id-time=2082787201l; ubid-main=135-8780951-6085065; x-wl-uid=1P7DnO8K9OhQvJKg7kHNecZi1CtzJUFzZjRCEFWtR7lfg3hXP3uheuXLnIEMpj7FJkTG0mKns97g=; session-token=AYoOHHbjvRn8LPuihez3O+w4FZ9ekQXZCnOgoptgP9K4cNb3LGU8rHXRpQkZoE0Mu2E8UUr8yL4sE4hwkGaqm+4iZJXA6btLjEzyxnW4Z44cL3BeSSuV4YRyHnA00LUETCizkNG1lbLH8akYAdLxHhqbA1CiEif/9zz0hL5oMY7grjMmfhU08MynlCntVuDV0T4ZIiz898n0W97nLKhz3TQRzbd6YkxUfkMbFBL2jyNuWnCHhYL6dXGC9I0/gRRL; aws-target-static-id=1552615676456-506063; aws-target-data=%7B%22support%22%3A%221%22%7D; aws-target-visitor-id=1552615676460-140811.22_27; aws-mkto-trk=id%3A112-TZM-766%26token%3A_mch-aws.amazon.com-1552615677500-93600; s_dslv=1554971524146; s_vn=1579768624363%26vn%3D6; s_nr=1554971524150-Repeat; skin=noskin; sp-cdn="L5Z9:HK"; lc-main=en_US; i18n-prefs=USD; csm-hit=tb:8DVK89TZ8DTZQ6BGG9N1+s-EBXG0YJKM83JB98EZ162|1560475702771&t:1560475702771&adb:adblk_yes'
        , 'Host': 'www.amazon.com'
        , 'Upgrade-Insecure-Requests': '1'
        , 'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.169 Safari/537.36'
    }
#    client = AsyncIOMotorClient(DB_HOST, DB_PORT)
#    db = client[DB_NAME][DB_COLLECTION]
    proxy = Proxy(True, False)
    semaphore = asyncio.Semaphore(multi)
    count = 20
    
    def _genkw():
        '''读入待查的关键字列表
        ==========================
        '''
        with codecs.open('keywords.txt', 'r', 'utf8') as f:
            temp = f.readline()
            while temp:
                yield temp
                temp = f.readline()
        
    async def _write2db(html, kw, page):
        '''请求完成后的处理，写入数据库
        =================================
        '''
        tree = etree.HTML(html)
        product_eles = getXpath(tree, r"//div[@class='sg-col-inner']/div/h2/a/span", firstOne=False)
        datas = map(lambda x: {'keyword': kw, 'product': getEleTxt(x, ""), 'page': page, 'batch_id': BATCH_ID}, product_eles)
        await db.insert_many(datas)
        logging.info('采集了 %s 第%s页%s个' % (kw, page, len(product_eles)))
    
    if proxy.useSockt5:
        connector = SocksConnector.from_url(proxy.getProxy()['http'])
        #async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [asyncRequest(r'https://www.amazon.com/s', semaphore, session = None, data = {'k': kw.strip(), 'page': str(page)}, \
                              headers = headers, returnHTML= True, retrySeconds = 33, callback= _write2db) for kw in _genkw() for page in range(1, 21)]
        await asyncio.wait(tasks)
    else:
        #async with aiohttp.ClientSession() as session:
        fileKws = list(_genkw())
        oneKws = 4
        for i in range(0, len(fileKws), oneKws):
            tasks = [asyncRequest(r'https://www.amazon.com/s', semaphore, session = None, data = {'k': kw.strip(), 'page': str(page)}, \
                              headers = headers, returnHTML= True, proxies= proxy, retrySeconds = 33, callback= _write2db) for page in range(1, 21) for kw in fileKws[i: i + oneKws]]
            await asyncio.wait(tasks)

def runfetch():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(fetchdata(40))
    logging.info('完成')






#def genMongo2df(columns = [], convertDatetime = [], condition = {}, cache = 1000):
#    '''从mongodb中读出到df
#    =======================
#    
#    参数
#    -------
#    **columns** 导出的列名
#    **convertDatetime**
#    **condition**
#    **cache** 每次处理的行数
#    '''
#    client = MongoClient(DB_HOST, DB_PORT)
#    db = client[DB_NAME][DB_COLLECTION]
#    count = db.count_documents(condition)
#    for i in range(count// cache + 1 if count % cache else 0):
#        data = list(db.find(condition, dict({k: 1 for k in columns}, **{'_id': 0}))[cache * i: min(cache * (i + 1), count)])
#        df = pd.DataFrame.from_dict(data)
#        if convertDatetime:
#            for c in convertDatetime:
#                df[c] = pd.to_datetime(df[c])
#        if columns:
#            for c in set(columns).difference(list(df.columns)): 
#                if c: df[c] = None
#        yield df
#
#def mongo2xls(condition= {}):
#    '''数据保存到excel
#    =====================
#    '''
#    create = True
#    #columns = getMongokeys(condition={'batch_id': BATCH_ID})
#    columns = ['keyword', 'product', 'page']
#    #for df in genMongo2df(columns= columns, condition= condition):
#        #df.to_csv(CONTACT_FILE, index = False, mode = 'w' if create else 'a')
#        #create = False  
#        #logging.info('数据保存到文件%s..' % CONTACT_FILE)
#    df = pd.concat(list(genMongo2df(columns= columns, condition= condition)), ignore_index=True, sort=False)
#    df.to_excel(CONTACT_FILE, index = False)
#    logging.info('数据保存到文件完成')
#
#def getMongokeys(condition = {}):
#    '''遍历出Mongo文档中的全部键值
#    ===============================
#    '''
#    client = MongoClient(DB_HOST, DB_PORT)
#    db = client[DB_NAME][DB_COLLECTION]
#    keys = []
#    for row in db.find(condition): keys = list(set(list(row.keys()) + keys))
#    return keys
#
#if __name__ == '__main__':
#    #runfetch()
#    mongo2xls()
#    