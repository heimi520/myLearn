#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:56:58 2019

@author: heimi
"""


import pandas as pd
import numpy as np
import os

root_amazon='../data/amazon_data'
root_ali='../data/ali_data'

def read_csv_list(root,source):
    line_list=[]
    for v in os.listdir(root):
        if 'csv' in v:
            print(v)
            path=os.path.join(root,v)
            td=pd.read_csv(path)
            
            if len(td)>1:
                line_list.append(td.values)
    dd_am=pd.DataFrame(np.row_stack(line_list),columns=['kw', 'mainProducts', 'number', 'productName'] if source=='ali'  else ['kw', 'productName', 'page'])
    dd_am=dd_am[['kw','productName']].drop_duplicates()
    dd_am=dd_am.rename(columns={'keyword':'PRODUCT_TAG_NAME','product':'PRODUCT_NAME'})
    dd_am['BUYSELL']='sell'
    dd_am['source']=source
    return dd_am


dd_am=read_csv_list(root_amazon,'amazon')
dd_ali=read_csv_list(root_ali,'ali')

data=pd.concat([dd_am,dd_ali],axis=0)
data=data.rename(columns={'productName':'PRODUCT_NAME','kw':'PRODUCT_TAG_NAME'})
data=data[data['PRODUCT_NAME'].notnull()]
data['PRODUCT_TAG_ID']=data['PRODUCT_TAG_NAME'].copy()


data=data[data['PRODUCT_TAG_NAME'].notnull()]
data['PRODUCT_TAG_ID']=data['PRODUCT_TAG_NAME'].copy()


###############################################################################
tag_map=data.set_index('PRODUCT_TAG_NAME')['PRODUCT_TAG_ID'].to_dict()

tag_map.update({ 
        'Camcorders':'Video Cameras',
        'Fitness Trackers':'Smart Watches',
        'Pantyhose / Tights':'Stockings',
        'Men Vests & Waistcoats':'Men Vests',
        'Children Vests & Waistcoats':'Children Vests',
        'Women Vests & Waistcoats':'Women Vests',
        'Smart Remote Control':'Remote Control',
        'Women Fur Coats':'Women Coats',
        'Men Fur Coats':'Men Coats',
        'Children Underwear':'Brief/Underwear',
        'PDAs':np.nan,
        'Printers':np.nan,
        'Tripods':np.nan,
        'Cleaners':np.nan,
        'Buckles':np.nan,
        'Electronic Books':np.nan,
        'Industrial Computer & Accessories':np.nan,
        'Hotel Uniforms':'Restaurant & Bar & Hotel & Promotion Uniforms',
        'Restaurant & Bar Uniforms':'Restaurant & Bar & Hotel & Promotion Uniforms',
        'Bank Uniforms':'Bank & Airline Uniforms',
        'Airline Uniforms':'Bank & Airline Uniforms',
        '3D Glasses':'VR & AR Glasses',
        'VR & AR':'VR & AR Glasses',
        'Fans & Cooling':'Fans & Cooling Pads',
        'Laptop Cooling Pads':'Fans & Cooling Pads',
        'Sewing Needles':'Sewing Needles & Threads',
        'Sewing Threads':'Sewing Needles & Threads',
        'Cassette Recorders & Players':'Cassette Players & Tapes',
        'Blank Records & Tapes':'Cassette Players & Tapes',
        'India & Pakistan Clothing':'Ethnic Clothing',
        'Asia & Pacific Islands Clothing':'Ethnic Clothing',
        'Africa Clothing':'Ethnic Clothing',
        'Traditional Chinese Clothing':'Ethnic Clothing',
        'Islamic Clothing':'Ethnic Clothing',
        
         'Laptop Bags & Cases	':'Laptop & PDA Bags & Cases',
         'PDA Bags & Cases':'Laptop & PDA Bags & Cases',
         'DVD Player Bags'	:'CD/DVD Player Bags & Cases',
         'CD Player Bags':'CD/DVD Player Bags & Cases',
         'VCD Player Bags':'CD/DVD Player Bags & Cases',
         'MP3 Bags & Cases':'MP3/MP4 Bags & Cases',
         'MP4 Bags & Cases':'MP3/MP4 Bags & Cases',
         'Home CD Players':'Home CD DVD & VCD Players',	
         'Home DVD Players':'Home CD DVD & VCD Players',
         'Home VCD Players':'Home CD DVD & VCD Players',
         'Portable CD Players	':'Portable CD DVD & VCD Players',
        'Portable DVD Players':'Portable CD DVD & VCD Players',
        'Wedding Jackets':	'Wedding Jackets / Wrap',
        'Wedding Wrap':'Wedding Jackets / Wrap',
         'Game Joysticks':'Joysticks & Game Controllers',
         'Game Controllers':'Joysticks & Game Controllers',
         
        })
    
        
data['PRODUCT_TAG_NAME']=data['PRODUCT_TAG_NAME'].map(tag_map)


data['PRODUCT_TAG_NAME']=data['PRODUCT_TAG_NAME'].map(tag_map)

data=data[data['PRODUCT_TAG_NAME'].notnull()]

tag1=pd.read_excel('../data/ali_data/服装&3C 标签7.02_new.xlsx',sheet_name=0)
tag2=pd.read_excel('../data/ali_data/服装&3C 标签7.02_new.xlsx',sheet_name=1)
tag=pd.concat([tag1,tag2],axis=0)
tag['PRODUCT_TAG_NAME']=tag['Product Tag'].str.replace('(^\s*)|(\s*$)','')
tag=tag[['PRODUCT_TAG_NAME','T1','T2']]
tag.columns=['PRODUCT_TAG_NAME','T1','T2']

md=pd.merge(data,tag,on=['PRODUCT_TAG_NAME'],how='left')

md['T1']=md['T1'].fillna('T1_Other')
md['T2']=md['T2'].fillna('T2_Other')


md.sample(10000).info()


cols_list=['PRODUCT_TAG_NAME','T1', 'T2','source','PRODUCT_NAME']
md[cols_list].to_csv('../data/input/apparel_3c_orig.csv',index=False)


md_sect=pd.merge(data,tag,on=['PRODUCT_TAG_NAME'],how='inner')



