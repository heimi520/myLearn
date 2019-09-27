#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 14:28:43 2019

@author: heimi
"""

import arrow
from email import encoders
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
import smtplib
import os

TITLE = 'Leads监控预警' + arrow.now().strftime('_%Y%m%d')
#CONTENT = '这里是推荐系统测试的log  请查看附件日志内容!'

def sendmsg(f_email, f_pwd, to_list, smtp_server, content,sendfile=None, is_seed=True):
    """
    sendmsg,使用smtplib发送邮件,这里使用网易邮箱作为发送方
    """
    
    # 邮件对象
    msg = MIMEMultipart()
    msg['From'] = f_email
    msg['Subject'] = TITLE
    to_str = ''
    for x in to_list:
        to_str += x + ','
    msg['To'] = to_str
    msg.attach(MIMEText(content,'plain','utf-8'))
    
#    sendfile_basename = os.path.basename(sendfile)

#    attachment = MIMEBase('application', 'octet-stream')
#    attachment.set_payload(open(sendfile, 'rb').read())
#    encoders.encode_base64(attachment)
#    attachment.add_header('Content-Disposition', 'attachment; filename="{}"'.format(sendfile_basename))
#    msg.attach(attachment)
        
    
    server = smtplib.SMTP()
    server.set_debuglevel(1)
    server.connect(smtp_server,25)
    server.starttls()
    server.login(f_email,f_pwd)
    server.sendmail(f_email,to_list,msg.as_string())
    server.quit()
    
    
    
    
    
#    
#DEVELOPERS = ['mijiaqi@meorient.com']
#
#content='test//////////////////'
#sendmsg('shujuguanli@meorient.com', 'Dc123456789', DEVELOPERS, 'smtp.exmail.qq.com',content)

#sendmsg(f_email, f_pwd, to_list, smtp_server, content,sendfile=None, is_seed=True):
#    """




    
    