
import pandas as pd


data=pd.read_excel('../data/meorient_data/付费买家数据翻译后打标.xlsx',encoding='gbk').rename(columns={'product':'PRODUCT_NAME'})

#tag_stand=pd.read_excel('../data/tagpack2/8.19 打标标签.xlsx',sheetname=0)
#tag_stand.columns=['PRODUCT_TAG_NAME','T1','T2']
#

#data=pd.merge(data,tag_stand[['T1']].drop_duplicates(),on=['T1'],how='inner')


md=data[['PRODUCT_NAME']].drop_duplicates()

md=md.rename(columns={'PRODUCT_NAME':'source_text'})
md['fromLang']='auto'
md['toLang']='en'

md.to_csv('../data/meorient_data/付费买家数据翻译后打标_need_transed.csv',index=False,encoding='utf8')



