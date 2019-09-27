

cd /home/heimi/文档/gitCodeLessData/myLearn/推荐/LEADS_demo


source  ~/.bashrc

while  true
do
  time=$(date "+%Y-%m-%d %H:%M:%S")
  echo ${time}
    
       
    while read line;do  
        eval "$line"  
    done < config.py  
    echo $channel

    if [ $channel == 'f' ]
    then
       echo "channel is factory"
        /home/heimi/anaconda3/bin/python    leads_from_factory.py
      echo "channel is factory ok"
    elif [ $channel == 'p' ]
    then
       echo "channel is producer"
        /home/heimi/anaconda3/bin/python    leads_from_producer.py
        echo "channel is producer ok"
    else
       echo "没有符合的条件"
    fi


  sleep 3
done
 
