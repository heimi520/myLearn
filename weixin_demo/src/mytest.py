#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 13:39:01 2019

@author: heimi
"""

import pandas as pd
import requests
import os


url_token='https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid=ww8de0ba62737429ce&corpsecret=JtL_ABS1d67jMkbCB0I6IS2ZrxHql5j5c5Flhmfjr5g'


res=requests.get(url_token)
token_json=res.json()

token=token_json['access_token']


url_department='https://qyapi.weixin.qq.com/cgi-bin/department/list?access_token=%s&id='%token

res=requests.get(url_department)
department_json=res.json()
part_pd=pd.DataFrame(department_json['department'])


#department	name	userid
#[1016011673]	米佳奇	mijiaqi


id_sjglb='1016011508'
url_detail='https://qyapi.weixin.qq.com/cgi-bin/user/simplelist?access_token=%s&department_id=%s&fetch_child=1'%(token,id_sjglb)

res=requests.get(url_detail)
detail_json=res.json()

detail_pd=pd.DataFrame(detail_json['userlist'])

user_id='mijiaqi'



url_label='https://qyapi.weixin.qq.com/cgi-bin/tag/list?access_token=%s'%token

res=requests.get(url_label)
label_json=res.json()['taglist']


tagid=1
url_label_detail='https://qyapi.weixin.qq.com/cgi-bin/tag/get?access_token=%s&tagid=%s'%(token,tagid)

tag_detail=requests.get(url_label_detail).json()


url_text='https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token=%s'%token

#requests.get(url_text).json()

#url_text='https://qyapi.weixin.qq.com/cgi-bin/message/send'
data={
   "touser" :"mijiaqi",
#   "toparty" :"",
#   "totag" : "",
   "msgtype":"text",
   "agentid" : 1000007,
#   "content":"test/////////",
   "text" : {
       "content":"test///// <a href=\"http://work.weixin.qq.com\">test//////</a>，test。"
   },
#   "safe":0,
#   "enable_id_trans": 0
}

#
#param={
#   "touser" : "UserID1|UserID2|UserID3",
#   "toparty" : "PartyID1|PartyID2",
#   "totag" : "TagID1 | TagID2",
#   "msgtype" : "text",
#   "agentid" : 1,
#   "text" : {
#       "content" : "你的快递已到，请携带工卡前往邮件中心领取。\n出发前可查看<a href=\"http://work.weixin.qq.com\">邮件中心视频实况</a>，聪明避开排队。"
#   },
#   "safe":0,
#   "enable_id_trans": 0
#}
import json
param = json.dumps(data, ensure_ascii=False)

res=requests.post(url_text,data=param.encode("utf-8").decode("latin1"))
print(res.json())












