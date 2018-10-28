# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 09:03:32 2018

@author: heimi
"""


print('git test///assssssssssssss//')
print('local testaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
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



################################################################################
dd=pd.read_csv('../data/input/基础数据2.csv',encoding='utf-8',parse_dates=['dateid'])
mydata=pd.read_csv('../data/temp/活动预测数据整合2.csv',parse_dates=['dateid','date_st','date_ed','sj_time'])

dd['item_status']=dd['item_status'].map({'已下架':1}).fillna(0)  
mydata['item_status']=mydata['item_status'].map({'已下架':1}).fillna(0) 

active_pd=mydata[mydata['label']=='activeFlag']
pre_pd=mydata[mydata['label']=='preFlag']

dd=pd.merge(dd,active_pd[['spu_ps','dateid','active_sell']],on=['spu_ps','dateid'],how='outer').fillna(0)
#dd['spu']=dd['spu_ps'].apply(lambda x:x[:-5])
#dd['month']=dd['dateid'].apply(lambda x:x.month)
#dd['year']=dd['dateid'].apply(lambda x:x.year)

aa=dd.groupby('dateid')['dateid'].count()



def get_unstack(md,ind):
    sd=md.set_index(['spu_ps','dateid'])[[ind]].unstack()
    sd.columns=sd.columns.get_level_values(level=1)
    sd=sd.T
    cols=sd.iloc[-1,:].sort_values(ascending=False).index.tolist()
    sd=sd[cols]
    return sd



aa17_99=dd[(dd['dateid']>=pd.to_datetime('20170909'))&(dd['dateid']<=pd.to_datetime('20170910'))]
aa18_99=dd[(dd['dateid']>=pd.to_datetime('20180909'))&(dd['dateid']<=pd.to_datetime('20180910'))]

aa17_normal=dd[(dd['dateid']>pd.to_datetime('20170910'))&(dd['dateid']<pd.to_datetime('20171020'))]
aa18_normal=dd[(dd['dateid']>pd.to_datetime('20180910'))&(dd['dateid']<pd.to_datetime('20181020'))]


aa17_db11=dd[(dd['dateid']>=pd.to_datetime('20171111'))&(dd['dateid']<=pd.to_datetime('20171111'))]


#
#qty17_99=get_unstack(aa17_99,'pay_number')
#uv17_99=get_unstack(aa17_99,'uv')
#car17_99=get_unstack(aa17_99,'Collection_number')
#save17_99=get_unstack(aa17_99, 'purchases_number')


def get_qty2uv(aa17_99):
    md17_99=aa17_99.groupby('mid_cls_name')[['pay_number','uv']].sum()
    md17_99.columns=['qtySum','uvSum']
    md17_99['qty2uv']=md17_99['qtySum']/md17_99['uvSum']
    md17_99['mid_cls_name']=md17_99.index
    return md17_99

md17_99=get_qty2uv(aa17_99)
md18_99=get_qty2uv(aa18_99)

md17_normal=get_qty2uv(aa17_normal)
md18_normal=get_qty2uv(aa18_normal)


md17_db11=get_qty2uv(aa17_db11)


def get_merge(md17_99,md17_db11):
    md99_db11=pd.merge(md17_99,md17_db11,on=['mid_cls_name'])
    md99_db11=md99_db11.sort_values('qty2uv_x',ascending=False)
    md99_db11=md99_db11[md99_db11>0]
    md99_db11=md99_db11[md99_db11.isnull().sum(axis=1)==0]
    md99_db11=md99_db11[md99_db11['qty2uv_x']<0.1]
    md99_db11=md99_db11[md99_db11['qty2uv_y']<0.1]
    return md99_db11


#md99_db11=get_merge(md17_99,md17_db11)
#md99_db11.plot.scatter('qty2uv_x','qty2uv_y')


###同比数据###################################
md99=get_merge(md17_99,md18_99)
md99['qty2uv_pct']=md99['qty2uv_y']/md99['qty2uv_x']

md_normal=get_merge(md17_normal,md18_normal)
md_normal['qty2uv_pct']=md99['qty2uv_y']/md99['qty2uv_x']


rd=pd.concat([md99,md_normal],axis=0)
#rd=rd[rd['qty2uv_x']<0.06]
rd=rd[['uvSum_x','uvSum_y']]
rd.columns=['x','y']

#####同比点图######################
#fig,ax=plt.subplots()
#md99.plot.scatter('uvSum_x','uvSum_y',ax=ax,c='r',s=40)
#md_normal.plot.scatter('uvSum_x','uvSum_y',ax=ax,c='g',s=40)
#ax.legend(['99 big sell','0911-1019'])
#ax.set_xlabel('17uv',fontproperties=zhfont,fontsize=10)
#ax.set_ylabel('18uv',fontproperties=zhfont,fontsize=10)
#fig.suptitle('2017-2018 中类　同比uv关系',fontproperties=zhfont,fontsize=20)
#

#md17_00to11=get_merge(md17_normal,md17_99)



#qudd99_pct=pd.concat([md17_99[['qty2uv']],md18_99[['qty2uv']]],axis=1)

###环比数据##############################
qudd17=pd.concat([md17_99[['uvSum']],md17_normal[['uvSum']],md17_db11[['uvSum']]],axis=1)
qudd17=qudd17[qudd17>100]
qudd17=qudd17[qudd17.isnull().sum(axis=1)==0]
qudd17.columns=['x1','x2','y']

qudd17[['x1','x2','y']].corr()

###环比图######################
#qudd17.plot()


####环比点图######################
#fig,ax=plt.subplots()
#qudd17.plot.scatter('x1','y',ax=ax,c='r',s=40)
#qudd17.plot.scatter('x2','y',ax=ax,c='g',s=40)
#ax.legend(['99 big sell','0911-1019'])
#ax.set_xlabel('17uv',fontproperties=zhfont,fontsize=10)
#ax.set_ylabel('17双十一uv',fontproperties=zhfont,fontsize=10)
#fig.suptitle('2017 中类　环比uv关系',fontproperties=zhfont,fontsize=20)
#
#



#############################3
qudd18=pd.concat([md18_99[['uvSum']],md18_normal[['uvSum']]],axis=1)
qudd18=qudd18[qudd18>100]
qudd18=qudd18[qudd18.isnull().sum(axis=1)==0]
qudd18.columns=['x1','x2']


from sklearn.linear_model import LinearRegression

###环比模型#################################
cols=['x1']
lr1_hb=LinearRegression(normalize=True,fit_intercept=True)
lr1_hb.fit(qudd17[cols],qudd17[['y']])
yhat=lr1_hb.predict(qudd17[cols])
qudd18['yhat1']=lr1_hb.predict(qudd18[cols])


cols=['x2']
lr2_hb=LinearRegression(normalize=True,fit_intercept=True)
lr2_hb.fit(qudd17[cols],qudd17[['y']])
qudd18['yhat2']=lr1_hb.predict(qudd18[cols])


cols=['x1','x2']
lr3_hb=LinearRegression(normalize=True,fit_intercept=True)
lr3_hb.fit(qudd17[cols],qudd17[['y']])
qudd18['yhat3']=lr3_hb.predict(qudd18[cols])

qudd18['yhat4']=qudd18[['yhat1','yhat2']].mean(axis=1)


ret18_db11_hb=qudd18[['yhat4']]
ret18_db11=ret18_db11_hb.iloc[1:]
ret18_db11['rate']=ret18_db11['yhat4']/ret18_db11['yhat4'].sum()
#ret18_db11=ret18_db11.astype(str)

ret18_db11.to_csv('uv.csv')

#ret18_db11['rate2']=ret18_db11['yhat2']/ret18_db11['yhat2'].sum()


#qudd=pd.concat([qudd17,qudd18],axis=1)

######同比模型###########################
#lr_tb=LinearRegression(normalize=True,fit_intercept=True)
#lr_tb.fit(rd[['x']],rd[['y']])
#yhat=lr_tb.predict(rd[['x']])
#rd['yhat']=yhat
#rd['mae']=(rd['y']-rd['yhat']).abs()
#rd['error']=rd['mae']/rd['y']
#
#rd['error'].mean()
#
#md17_db11['yhat_tb']=lr_tb.predict(md17_db11[['qty2uv']])
#
#ret18_db11_tb=md17_db11[['yhat_tb']]





#md17_00to11.plot.scatter('qty2uv_x','qty2uv_y')
#md17_00to11[['qty2uv_x','qty2uv_y']].corr()















#
#md99=pd.merge(md17_99,md18_99,on=['mid_cls_name'])
#md99=md99.sort_values('qty2uv_x',ascending=False)
#md99=md99[md99>0]
#md99=md99[md99.isnull().sum(axis=1)==0]
#md99[['qty2uv_x','qty2uv_y']].corr()




#md99.plot.scatter('qty2uv_x','qty2uv_y')
























