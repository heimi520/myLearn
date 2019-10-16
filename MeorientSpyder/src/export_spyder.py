#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:19:30 2019

@author: heimi
"""

import pandas as pd
import requests
from lxml import etree
import time

url_home='http://wmsw.mofcom.gov.cn/wmsw/'
#res=requests.get(url_home)
#html=res.text

#with open('../data/input/home.html','a+') as f:
#    f.write(html)
with open('../data/input/home.html','r') as f:
    html=f.read()


def xpath_html(html):
    tree=etree.HTML(html)
    line_list=[]
    for a in tree.xpath('//div[@class="keyList clearfix"]//li/a'):
        codeclass=str(a.xpath('./@codeclass')[0])
        text=a.xpath('./text()')[0].replace('\xa0','')
        line_list.append([codeclass,text])
    return line_list




url_class='http://wmsw.mofcom.gov.cn/wmsw/getCodeByCodeClass'
url_father='http://wmsw.mofcom.gov.cn/wmsw/getCodeByFather'


def get_sub(codeclass,cname,country_code):
    param={'codeClass':codeclass}
    res=requests.post(url_class,data=param)
    text_class=res.json()
    ret_list=[]
    for k,line in enumerate(text_class):
        print(k)
        fatherCode=line[0]  
#        lang=line[3]
        class_name=line[1]
        param={'fatherCode':fatherCode,
               'queryType':'CN',
               'hsCountry':country_code,
               'keyWord':''
               }
        headers_class={
                'Accept': '*/*',
                'Accept-Encoding': 'gzip, deflate',
                'Accept-Language': 'zh-CN,zh;q=0.9',
                'Connection': 'keep-alive',
                'Content-Length': '11',
                'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
#                'Cookie': 'JSESSIONID=CAA29D75978C61785051A380E997DFC6; insert_cookie=40867542',
                'Host': 'wmsw.mofcom.gov.cn',
                'Origin': 'http://wmsw.mofcom.gov.cn',
                'Referer': 'http://wmsw.mofcom.gov.cn/wmsw/',
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36',
                'X-Requested-With': 'XMLHttpRequest'
                                
            }    
                
        
        
        res=requests.post(url_father,headers=headers_class,data=param)
        code_json=res.json()
        code_pd=pd.DataFrame(code_json,columns=['code','name','col1','col2'])
        code_pd['fatherCode']=fatherCode
        code_pd['className']=class_name  
        code_pd['codeClass']=codeclass
        code_pd['codeName_NUM']=cname

        ret_list.append(code_pd)
    ret_pd=pd.concat(ret_list,axis=0)
    return ret_pd





country_list=[["AD","安道尔"],["AE","阿联酋"],["AF","阿富汗"],["AG","安提瓜和巴布达"],["AL","阿尔巴尼亚"],["AM","亚美尼亚"],["AO","安哥拉"],["AR","阿根廷"],["AT","奥地利"],["AU","澳大利亚"],["AZ","阿塞拜疆"],["BA","波黑"],["BB","巴巴多斯"],["BD","孟加拉国"],["BE","比利时"],["BF","布基纳法索"],["BG","保加利亚"],["BH","巴林"],["BI","布隆迪"],["BJ","贝宁"],["BN","文莱"],["BO","玻利维亚"],["BR","巴西"],["BS","巴哈马"],["BT","不丹"],["BW","博茨瓦纳"],["BY","白俄罗斯"],["BZ","伯利兹"],["CA","加拿大"],["CD","刚果（金）"],["CF","中非"],["CG","刚果（布）"],["CH","瑞士"],["CI","科特迪瓦"],["CK","库克群岛"],["CL","智利"],["CM","喀麦隆"],["CN","中国"],["CO","哥伦比亚"],["CR","哥斯达黎加"],["CU","古巴"],["CV","佛得角"],["CY","塞浦路斯"],["CZ","捷克"],["DE","德国"],["DJ","吉布提"],["DK","丹麦"],["DM","多米尼克"],["DO","多米尼加"],["DZ","阿尔及利亚"],["EC","厄瓜多尔"],["EE","爱沙尼亚"],["EG","埃及"],["ER","厄立特里亚"],["ES","西班牙"],["ET","埃塞俄比亚"],["FI","芬兰"],["FJ","斐济"],["FM","密克罗尼西亚联邦"],["FR","法国"],["GA","加蓬"],["GB","英国"],["GD","格林纳达"],["GE","格鲁吉亚"],["GH","加纳"],["GM","冈比亚"],["GN","几内亚"],["GQ","赤道几内亚"],["GR","希腊"],["GT","危地马拉"],["GW","几内亚比绍"],["GY","圭亚那"],["HK","中国香港"],["HN","洪都拉斯"],["HR","克罗地亚"],["HT","海地"],["HU","匈牙利"],["ID","印度尼西亚"],["IE","爱尔兰"],["IL","以色列"],["IN","印度"],["IQ","伊拉克"],["IR","伊朗"],["IS","冰岛"],["IT","意大利"],["JM","牙买加"],["JO","约旦"],["JP","日本"],["KE","肯尼亚"],["KG","吉尔吉斯斯坦"],["KH","柬埔寨"],["KI","基里巴斯"],["KM","科摩罗"],["KN","圣基茨和尼维斯"],["KP","朝鲜"],["KR","韩国"],["KW","科威特"],["KZ","哈萨克斯坦"],["LA","老挝"],["LB","黎巴嫩"],["LC","圣卢西亚"],["LI","列支敦士登"],["LK","斯里兰卡"],["LR","利比里亚"],["LS","莱索托"],["LT","立陶宛"],["LU","卢森堡"],["LV","拉脱维亚"],["LY","利比亚"],["MA","摩洛哥"],["MC","摩纳哥"],["MD","摩尔多瓦"],["ME","黑山"],["MG","马达加斯加"],["MH","马绍尔群岛"],["MK","北马其顿"],["ML","马里"],["MM","缅甸"],["MN","蒙古"],["MO","中国澳门"],["MR","毛里塔尼亚"],["MT","马耳他"],["MU","毛里求斯"],["MV","马尔代夫"],["MW","马拉维"],["MX","墨西哥"],["MY","马来西亚"],["MZ","莫桑比克"],["NA","纳米比亚"],["NE","尼日尔"],["NG","尼日利亚"],["NI","尼加拉瓜"],["NL","荷兰"],["NO","挪威"],["NP","尼泊尔"],["NR","瑙鲁"],["NU","纽埃"],["NZ","新西兰"],["OM","阿曼"],["PA","巴拿马"],["PE","秘鲁"],["PG","巴布亚新几内亚"],["PH","菲律宾"],["PK","巴基斯坦"],["PL","波兰"],["PR","波多黎各"],["PS","巴勒斯坦"],["PT","葡萄牙"],["PW","帕劳"],["PY","巴拉圭"],["QA","卡塔尔"],["RO","罗马尼亚"],["RS","塞尔维亚"],["RU","俄罗斯"],["RW","卢旺达"],["SA","沙特阿拉伯"],["SB","所罗门群岛"],["SC","塞舌尔"],["SD","苏丹"],["SE","瑞典"],["SG","新加坡"],["SI","斯洛文尼亚"],["SK","斯洛伐克"],["SL","塞拉利昂"],["SM","圣马力诺"],["SN","塞内加尔"],["SO","索马里"],["SR","苏里南"],["SS","南苏丹"],["ST","圣多美和普林西比"],["SV","萨尔瓦多"],["SY","叙利亚"],["SZ","斯威士兰"],["TD","乍得"],["TG","多哥"],["TH","泰国"],["TJ","塔吉克斯坦"],["TL","东帝汶"],["TM","土库曼斯坦"],["TN","突尼斯"],["TO","汤加"],["TR","土耳其"],["TT","特立尼达和多巴哥"],["TV","图瓦卢"],["TW","中国台湾"],["TZ","坦桑尼亚"],["UA","乌克兰"],["UG","乌干达"],["US","美国"],["UY","乌拉圭"],["UZ","乌兹别克斯坦"],["VA","梵蒂冈"],["VE","委内瑞拉"],["VN","越南"],["VU","瓦努阿图"],["WS","萨摩亚"],["YE","也门"],["ZA","南非"],["ZM","赞比亚"],["ZW","津巴布韦"]]
country_pd=pd.DataFrame(country_list,columns=['code','name'])

spyder_pd=pd.read_excel('../data/input/爬取国家.xlsx')
spyder_pd=spyder_pd.iloc[:,[0]]
spyder_pd.columns=['name']
spyder_pd['flag']=1


spyder_pd=pd.merge(spyder_pd,country_pd,on=['name'],how='left')




def crawl_country_data(country_code,country_name):
    code_list=xpath_html(html)
    ret_list=[]
    for k,(codeclass ,classname) in enumerate(code_list):
        print('%s/%s'%(k,len(code_list)), codeclass,classname)
        sub_pd=get_sub(codeclass,classname,country_code)
        ret_list.append(sub_pd)
        time.sleep(1)
     
    all_pd=pd.concat(ret_list,axis=0)
    all_pd.to_excel('../data/output/%s_%s_export.xlsx'%(country_name, country_code),encoding='gbk',index=False)


 
    
for (code,name) in spyder_pd[['code','name']].values.tolist():
    print(code,name)
    try:
        crawl_country_data(code,name)
    except Exception as e:
        print(code,name,e)
#    break    
    


















