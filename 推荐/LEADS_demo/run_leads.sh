
source  ~/.bashrc
cd /home/heimi/文档/gitCodeLessData/myLearn/推荐/LEADS_demo


while  true
do
  time=$(date "+%Y-%m-%d %H:%M:%S")
  echo ${time}
  /home/heimi/anaconda3/bin/python    step2_leads_from_factory.py
  
  sleep 3
done
 
