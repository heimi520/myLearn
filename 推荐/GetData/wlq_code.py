#coding:utf8
import time, logging
import requests, base64
from functools import reduce
from sqlalchemy import create_engine
import cx_Oracle

'''同步数据工厂的数据
=========================
由api到本机oracle
'''
logging.basicConfig(level = logging.INFO, format='%(asctime)s %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

def getRequest(url, headers = None, method = 'get', data = None, proxies = None, timeout = 120, returnType = 'xml', retrySeconds = 1):
    '''请求一个url返回xlm树
    =====================================
    
    参数
    --------
    **returnType** 返回的样式：html, xml, json
    '''
    proxies = proxies.getProxy() if proxies is not None else None
    while True:
        try: 
            if method == 'get':
                r = requests.get(url, headers = headers, params= data, proxies = proxies, timeout = timeout)
            elif method == 'post':
                r = requests.post(url, headers = headers, data= data, proxies = proxies, timeout = timeout)
            if r.status_code == 200:
                if returnType == 'html': return r, True
                elif returnType == 'xml': return etree.HTML(r.text), True
                elif returnType == 'json': return r.json(), True
            else:
                logging.error('ERR')
                return r.status_code, False
        except Exception as e:
            logging.error(e)
            time.sleep(retrySeconds)
    return r.status_code, False

def fetchAttr(obj:dict, attr:str):
    '''取出点分的属性名
    ===================
    当对象是列表时，可以用数值指代第几个元素，以0为起点
    '''
    if obj is None: return 
    attrs = attr.split('.')
    param = [obj]
    param.extend(attrs)
    
    def _handle(x, y):
        if y.isdigit() and (isinstance(x, list) or isinstance(x, tuple)) and len(x) > int(y):
            return x[int(y)]
        elif y.isdigit() and (isinstance(x, list) or isinstance(x, tuple)) and len(x) <= int(y):
            return None
        else:
            return x.get(y, None)
        
    result = reduce(_handle, param)
    if result == {}: return None
    return result

def data2oracle():
    '''从网页接口访问数据写入到oracle
    ====================================
    同步方式，速率较低
    '''
#     ipAddr = '52.29.102.35'
    ipAddr = '172.31.7.119'
    dbPwd = base64.b64decode(b'TUVPQjJCaHM3eTNibkgjRzdHMjNWQg==').decode('utf8')
    #oracleEngine = create_engine('oracle+cx_oracle://MEORIENTB2B_BI:%s@%s:1521/ORCL?charset=UTF-8' % (dbPwd, ipAddr), echo=False, encoding='utf8')
    #oracleEngine.execute('create table factory(email1 varchar2(500) ,emailValid1 varchar2(500) ,emailValidResult1 varchar2(500) ,phone varchar2(500) ,phoneValid1 varchar2(500) ,mobile varchar2(500) ,mobileValid1 varchar2(500), writedate TIMESTAMP)')
    pagesize = 400
    ignoreRows = 0
    page = 1
    #到目前为止认为已同步的数据量，包含即将插入的量
    currentRows = 0
    # 一次插入数据库的行数
    insertednum = 4000
    currTime = int(time.time()*1000)
#     currTime = 0

    def _getTimeStart():
        '''获取上次同步的时间
        =========================
        '''
        oracleEngine = create_engine('oracle+cx_oracle://MEORIENTB2B_BI:%s@%s:1521/ORCL?charset=UTF-8' % (dbPwd, ipAddr), echo=False, encoding='utf8')
        cursor = oracleEngine.execute("select lasttime from factory_flag where rownum = 1")
        row = cursor.fetchone()
        if row is not None:
            return 0 if row[0] is None else row[0] 
        else:
            return 0
    
    def _setTimeStart():
        '''更新本次同步的时间
        =========================
        '''
        oracleEngine = create_engine('oracle+cx_oracle://MEORIENTB2B_BI:%s@%s:1521/ORCL?charset=UTF-8' % (dbPwd, ipAddr), echo=False, encoding='utf8')
        cursor = oracleEngine.execute("update factory_flag set lasttime = :1 where rownum = 1", (currTime, ))
        
    timeStart = _getTimeStart()
    #清空数据
    #cursor.execute('truncate table factory')
    waiteddata = []
    
    def _insert(data: list, currentRows: int):
        '''插入到数据库
        ===================
        
        参数
        ------
        **data** 即将插入Oracle的数据
        **currentRows** 即将记录到oracle，当前已完成同步的总数据量
        '''
        while True:
            db = None
            try:
                db = cx_Oracle.connect('MEORIENTB2B_BI/%s@%s:1521/ORCL' % (dbPwd, ipAddr), encoding = 'UTF8')
                cursor = db.cursor()
                #cursor.execute("update factory_flag set currentrow = :1, updatedate = current_timestamp where rownum = 1", (currentRows, ))
                cursor.execute("truncate table factory_work2")
                cursor.prepare("insert into factory_work2(opTime, monid,email1, emailValid1, emailValidResult1, phone, phoneValid1, mobile, mobileValid1, writedate) values (:opTime, :monid, :email1, :emailValid1, :emailValidResult1, :phone, :phoneValid1, :mobile, :mobileValid1, CURRENT_TIMESTAMP)")
                cursor.executemany(None, data)
                db.commit()
                sqls = [
                    'delete factory_work2 where rowid in (select min(rowid) rd from factory_work2 group by email1 having count(email1) > 1)',
                '''update factory a set (opTime, monid,email1, emailValid1, emailValidResult1, phone, phoneValid1, mobile, mobileValid1, writedate)
                = (select b.opTime, b.monid,b.email1, b.emailValid1, b.emailValidResult1, b.phone, b.phoneValid1, b.mobile, b.mobileValid1, CURRENT_TIMESTAMP from factory_work2 b where a.email1 = b.email1 and rownum = 1)
                where exists (select 1 from factory_work2 b where a.email1 = b.email1)
                ''',
                'delete factory_work2 b where exists (select 1 from factory a where a.email1 = b.email1)',
                'insert into factory(opTime, monid,email1, emailValid1, emailValidResult1, phone, phoneValid1, mobile, mobileValid1, writedate) select distinct b.opTime, b.monid,b.email1, b.emailValid1, b.emailValidResult1, b.phone, b.phoneValid1, b.mobile, b.mobileValid1, CURRENT_TIMESTAMP from factory_work2 b'
                ]
                rowcounts = []
                for sql in sqls:
                    cursor.execute(sql)
                    rowcounts.append(cursor.rowcount)
                db.commit()
                logging.info('执行一段DB操作，本批次数据%d行，删除重复%d行，更新%d行，插入%d行' % (len(data), rowcounts[0], rowcounts[1], rowcounts[3]))
                break
            except Exception as e:
                logging.error('%s， 重试' % e)
            finally:
                try:
                    if db is not None:
                        db.rollback()
                        cursor.close()
                        db.close()
                except Exception as e:
                    logging.error('关闭库时出错 %s' % e)
                
    while True:
        params = {'page': page, 'size': pagesize, 'timeStart': timeStart}
        if currTime > 0: params['timeEnd'] = currTime
        data, result = getRequest(r'http://40.89.189.173:9098/service/factorydata/queryPurchaseSourceByOptimePage', data = params, returnType = 'json')
        if result:
            if not data.get('success', False): 
                logging.error('访问第%d页时出错，重试' % page)
                continue
            else:
                page += 1
                dlist = fetchAttr(data, 'payload.ret')
                if isinstance(dlist, list) and len(dlist) > 0:
                    infos = [{'opTime': row.get('opTime', None), 'monid': row.get('id', None), 'email1': row.get('email1', None), 'emailValid1': row.get('emailValid1', None), 'emailValidResult1': row.get('emailValidResult1', None), 'phone': row.get('phone', None), 'phoneValid1': row.get('phoneValid1', None), 'mobile': row.get('mobile', None), 'mobileValid1': row.get('mobileValid1', None)} for row in dlist]
                    currentRows = (page - 2) * pagesize + len(infos)
                    logging.info('读完第%d页%d条数据' % ((page - 1), len(infos)))
                    waiteddata.extend(infos)
                    if len(waiteddata) >= insertednum:
                        _insert(waiteddata, currentRows)
                        waiteddata = []
                    #logging.info('同步完第%d页%d条数据' % ((page - 1), len(infos)))
                else:
                    _insert(waiteddata, currentRows)
                    waiteddata = []
                    break
    _setTimeStart()
    logging.info('同步完成')
                
if __name__ == '__main__':
    logging.info('开始...')
    data2oracle()
