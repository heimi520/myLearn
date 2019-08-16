#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:54:55 2019

@author: heimi
"""

import cx_Oracle 
db = cx_Oracle.connect('username/password@host')
print(db.version)
db.close()