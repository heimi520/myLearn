#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 14:47:00 2019

@author: heimi
"""


import json
import requests
import datetime

postUrl = 'https://gta1.growingio.com/_private/v5/projects/GRwd0KlP/chartdata'
# payloadData数据
payloadData ={"attrs":{"metricType":"none","subChartType":"seperate"},
              "dimensions":[],"filter":'null',
              "granularities":[{"id":"tm","interval":86400000}],
              "limit":20,
              "metrics":[{"id":"uc","name":"用户量","type":"prepared","platforms":["all"]},
                          {"id":"pv","name":"页面浏览量","type":"prepared","platforms":["all"]},
                          {"id":"vs","name":"访问量","type":"prepared","platforms":["all"]},
                          {"id":"pvpv","name":"每次访问页面浏览量",
                           "type":"prepared","platforms":["all"]}],
                "orders":[{"isDim":'false',"index":0,"orderType":"desc"}],
                "timeRange":"day:8,1","targetUser":"nPNYeVXR","skip":0}

# 请求头设置
#payloadHeader = {
#    'Host': 'sellercentral.amazon.com',
#    'Content-Type': 'application/json',
#}

payloadHeader = {
'accept': 'application/json',
'Accept-Language': 'zh',
'Content-Type': 'application/json',
'Origin': 'https://www.growingio.com',
'Referer': 'https://www.growingio.com/projects/GRwd0KlP/eventAnalysis/xogJay4P?timeRange=day%3A8%2C1&interval=86400000',
'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.86 Safari/537.36'
}
# 下载超时

timeOut = 25
# 代理


r = requests.post(postUrl, data=json.dumps(payloadData), headers=payloadHeader)



#dumpJsonData = json.dumps(payloadData)











