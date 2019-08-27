#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:56:58 2019

@author: heimi
"""


import pandas as pd
import numpy as np


am1=pd.read_csv('../data/amazon_data/amazon_com_products 20190614.csv')
am2=pd.read_csv('../data/amazon_data/amazon_com_products 20190620.csv')
am3=pd.read_csv('../data/amazon_data/amazon_com_products 20190702.csv')
am4=pd.read_csv('../data/amazon_data/amazon_com_products 20190703.csv')
am5=pd.read_csv('../data/amazon_data/amazon_com_products 20190709.csv')
am6=pd.read_csv('../data/amazon_data/amazon_com_products 20190710.csv')
dd_am=pd.concat([am1,am2,am3,am4,am5,am6],axis=0)

dd_am=dd_am[['keyword', 'product']].drop_duplicates()
dd_am=dd_am.rename(columns={'keyword':'PRODUCT_TAG_NAME','product':'PRODUCT_NAME'})
dd_am['PRODUCT_TAG_ID']=dd_am['PRODUCT_TAG_NAME'].copy()
dd_am['BUYSELL']='sell'
dd_am['source']='amazon'


ali0=pd.read_csv('../data/ali_data/alibaba_products.csv')
ali1=pd.read_csv('../data/ali_data/alibaba_products 20190624.csv')
ali2=pd.read_csv('../data/ali_data/alibaba_products 20190625.csv')
ali3=pd.read_csv('../data/ali_data/alibaba_products 20190702.csv')
ali4=pd.read_csv('../data/ali_data/alibaba_products 20190703.csv')
ali5=pd.read_csv('../data/ali_data/alibaba_products 20190709.csv')
ali6=pd.read_csv('../data/ali_data/alibaba_products 20190710.csv')
dd_ali=pd.concat([ali0,ali1,ali2,ali3,ali4,ali5,ali6],axis=0)


dd_ali=dd_ali[['kw', 'productName']].drop_duplicates()
dd_ali=dd_ali.rename(columns={'kw':'PRODUCT_TAG_NAME','productName':'PRODUCT_NAME'})
dd_ali['PRODUCT_TAG_ID']=dd_ali['PRODUCT_TAG_NAME'].copy()
dd_ali['BUYSELL']='sell'
dd_ali['source']='ali'


#dd2['PRODUCT_TAG_NAME'].unique().tolist()

data=pd.concat([dd_am,dd_ali],axis=0)
data=data[data['PRODUCT_NAME'].notnull()]

data.to_csv('../data/input/data_all_orig.csv',index=False)


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

data=data[data['PRODUCT_TAG_NAME'].notnull()]
data['PRODUCT_TAG_ID']=data['PRODUCT_TAG_NAME'].copy()

tag1=pd.read_excel('../data/ali_data/服装&3C 标签7.02_new.xlsx',sheet_name=0)
tag2=pd.read_excel('../data/ali_data/服装&3C 标签7.02_new.xlsx',sheet_name=1)
tag=pd.concat([tag1,tag2],axis=0)
tag['PRODUCT_TAG_NAME']=tag['Product Tag'].str.replace('(^\s*)|(\s*$)','')

#'Garment Cords' in set(tag['PRODUCT_TAG_NAME'])

data_filter=pd.merge(data,tag,on=['PRODUCT_TAG_NAME'],how='inner')

    
    
aa_count=data_filter.groupby('PRODUCT_TAG_NAME')['PRODUCT_TAG_NAME'].count().sort_values(ascending=True).to_frame('count')
aa_count['tag']=aa_count.index


cols_list=['PRODUCT_TAG_NAME','T1', 'PRODUCT_NAME', 'PRODUCT_TAG_ID', 'BUYSELL',
       'source']

data_filter[cols_list].to_csv('../data/input/data_filter.csv',index=False)












    
    

