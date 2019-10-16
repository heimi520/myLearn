#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:53:25 2019

@author: heimi
"""

import pandas as pd
import json
dd=pd.read_csv('tian_lin_jie_data_json.csv',index_col=0)


for v in dd['result'].tolist():
    line=v.replace("'",'"')
    line=line.replace('None','null')
    
    
    line="""
     
   {"name": "", "location": null, "twitter": null, "linkedin": null, "facebook": null, "bio": "", "logo": "", "website": null, 
   "founded": null, "employees": null, "locale": "en", "category": "Other", "details": {"locales": [{"code": "en", "name": "English"}], 
   "categories": [{"code": "OTHER", "name": "Other"}], "industries": [], "emails": [], "phones": [], "profiles": {}, "locations": [],
   "images": [], "urls": [], "keywords": [], "keyPeople": [] "updated": "2019-08-09"}
    
    }
    
    """
    line2=json.loads(line)
    
     "keyPeople": [], "traffic": {"countryRank": {"global": {"rank": 13614259, "name": "Global"}},
    "localeRank": {}}}, 
    "dataAddOns": [{"id": "keypeople", "name": "Key People", "enabled": False, 
    "applied": False, "description": "Displays information about people of interest at this company.", 
    "docLink": "http://docs.fullcontact.com/api/#key-people"}], "updated": "2019-08-09"
    
#    
#    , "bio": "", "logo": "", "website": 'null', "founded": 'null',
#    "employees": 'null', "locale": "en", "category": "Other", 
#    "details": {"locales": [{"code": "en", "name": "English"}],
#    "categories": [{"code": "OTHER", "name": "Other"}], 
#    "industries": [], "emails": [], "phones": [], "profiles": {}, 
#    "locations": [], "images": [], "urls": [], "keywords": [], 
#    "keyPeople": [], "traffic": {"countryRank": {"global": {"rank": 13614259, 
#    "name": "Global"}}, "localeRank": {}}}, "dataAddOns":
#       '', 
#            "updated": "2019-08-09"}
#    
#     [{"id": "keypeople", "name": "Key People", "enabled": False, 
#        "applied": False, "description":
#            "Displays information about people of interest at this company.",
#            "docLink": "http://docs.fullcontact.com/api/#key-people"}]
#    
#    