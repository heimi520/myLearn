# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:05:29 2019

@author: Administrator
"""

#watch -n 10 nvidia-smi
from selenium import webdriver
import time

import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
driver = webdriver.Chrome()
#如果phantomjs.exe没有设置全局变量，用以下语句
#driver=webdriver.PhantomJS(executable_path='你的phantomjs.exe路径')
#打开谷歌翻译
driver.get('https://translate.google.cn/')
#获取文本输入域
text_dummy=driver.find_element_by_class_name('tlid-source-text-input')
text_dummy.clear()
#输入 hello hangzhou
text_dummy.send_keys('hello hangzhou')
time.sleep(1)

text_translation=driver.find_element_by_xpath('//span[@class="tlid-translation translation"]/span')
#输入翻译结果
print(text_translation.text)
driver.close()

