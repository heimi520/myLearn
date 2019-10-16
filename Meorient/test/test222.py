#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:06:01 2019

@author: heimi
"""

from google.cloud import translate
translate_client = translate.Client()
 
text = u'hello,world'
target = 'ru'
 
translation = translate_client.translate(text,target_language=target)
 
