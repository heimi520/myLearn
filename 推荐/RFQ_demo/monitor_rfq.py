
import re
import os
import pandas as pd

root='logs'
path_list=[]
for v  in os.listdir(root):
    path=os.path.join(root,v)
    path_list.append(path)

path_list.sort()
path_last=path_list[-1]

remove_list=path_list[:-2]
try:
    for file in remove_list:
        os.remove(file)
except Exception as e:
    print('remove file error',e)

with open(path_last,'r') as f:
    dd=f.read()

block_list='\n'.join(dd.split('start block')[-5:])

orig_num=re.findall('[0-9\s+\-:]{10,19} INFO original factory lines:[0-9]{1,100}\n',block_list)
#filter_num=re.findall('[0-9\s+\-:]{10,19} INFO filter factory lines:[0-9]{1,100}\n',block_list)
#extract_num=re.findall('[0-9\s+\-:]{10,19} INFO extract factory lines:[0-9]{1,100}\n',block_list)
match_num=re.findall('[0-9\s+\-:]{10,19} INFO match all lines:[0-9]{1,100}\n',block_list)


if len(orig_num)>=2:
    line_list=[]
    for v1,v2 in zip(orig_num,match_num):
        tm=v1[:19]
        num1=v1.replace('\n','').split(':')[-1]
        num2=v2.replace('\n','').split(':')[-1]
        line_list.append([tm,num1,num2])
    
    line_pd=pd.DataFrame(line_list,columns=['dt','orig_num','match_total'])
    line_pd=line_pd.set_index('dt')
    line_pd=line_pd.astype(float)
    line_pd['match_num']=line_pd['match_total'].diff(1)
    #line_pd=line_pd.tail(1)
    










