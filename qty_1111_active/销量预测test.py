# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 17:52:48 2018

@author: heimi
"""


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
zhfont = mpl.font_manager.FontProperties(fname='/usr/share/fonts/truetype/droid/simhei.ttf')
plt.rcParams['font.family']= 'SimHei' # 解决中文乱码
plt.rcParams['axes.unicode_minus'] = False # 解决负号乱码
import pylab as pl

uv=pd.read_csv('uv.csv',index_col=0)
qty2uv=pd.read_csv('qty2uv.csv',index_col=0)

aa=pd.concat([uv,qty2uv],axis=1)
aa['qtyPred']=aa['yhat4']*aa['yhat']
aa=aa[['yhat4','yhat','qtyPred']]
aa=aa.sort_values('qtyPred',ascending=False)
aa['rate']=aa['qtyPred']/aa['qtyPred'].sum()
aa.columns=['预测uv','预测转化','预测销量','占比']
aa.to_csv('qty1111.csv')
aa=aa.astype(str)












