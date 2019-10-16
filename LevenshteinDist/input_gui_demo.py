#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:26:46 2019

@author: heimi
"""

import tkinter as tk




import pandas as pd
import numpy as np
import Levenshtein


def clean(line):
    line=line.lower().replace(' ','')
    return line
#dd=pd.read_excel('标签汇总-new.xlsx')
#dd[['TAG']].to_csv('tag.csv',index=False)

tag=pd.read_csv('tag.csv')
tag['tag_list']=tag['TAG'].apply(lambda x:x.split(' '))

line_list=[]
for tag_name,tag_list in tag[['TAG','tag_list']].values:
    for word in tag_list:
        line_list.append([word,tag_name])  
line_pd=pd.DataFrame(line_list,columns=['word','TAG'])
orig_pd=tag[['TAG']]
orig_pd['word']=orig_pd['TAG']
line_pd=pd.concat([orig_pd,line_pd],axis=0)

tag=line_pd.copy()
tag['dest_clean']=tag['word'].apply(clean)
tag['tag_clean']=tag['TAG'].apply(clean)
tag['len']=tag['tag_clean'].apply(lambda x:len(x))

def check(tag,src_line):
    import time
    t1=time.time()
    tag_list=tag['dest_clean'].tolist()
    dist_list=[]
    src=clean(src_line)
    len_src=len(src)
    for dest_line in  tag_list:
        dist=Levenshtein.distance(src,dest_line[:len_src])
        dist_list.append(dist)
    tag['src']=src_line
    tag['dist']=dist_list 

    tag=tag[tag['dist']<=0]
    
    tag=tag.sort_values('dist')
    sort_list=[]
    dist_list=list(set(tag['dist']))
    for dist in dist_list:
        td=tag[tag['dist']==dist]
        td=td.sort_values('len')
        sort_list.append(td)
    
    show_list=[]
    if len(sort_list)>0:
        sort_pd=pd.concat(sort_list,axis=0)  
        show_list=sort_pd.head(100)['TAG'].unique().tolist()
    t2=time.time()
    print('takes time',t2-t1)
    return show_list






def check_spell(e):
    global text
    var = entry.get()		#调用get()方法，将Entry中的内容获取出来
    
    check_line=check(tag,var)
    text.delete('1.0','end')
    for line in check_line:
        text.insert(tk.END, line+'\n')
#    print(var)

window = tk.Tk()
#frame = tk.Frame(window, width=100, height=500)
entry = tk.Entry(window,bd=2,width=100)
entry.bind("<KeyRelease>", check_spell)
entry.pack()

text = tk.Text(window, height=50, width=50)
text.pack()





#text = tk.Text(window, wrap="word", background="white", 
#                            borderwidth=0, highlightthickness=0)
#
#
#
#text.pack()
#def change_state():
#    var = entry.get()		#调用get()方法，将Entry中的内容获取出来
#    print(var)
#button = tk.Button(window,text='单击',command=change_state).pack()

window.mainloop()



