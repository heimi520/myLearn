# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:02:15 2019

@author: Administrator
"""


import googletrans
from googletrans import Translator
translator = Translator(service_urls=['translate.google.cn'])
source = '我还是不开心！'
text = translator.translate(source,src='zh-cn',dest='en').text
print(text)
