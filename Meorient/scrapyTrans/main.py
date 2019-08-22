# -*- coding: utf-8 -*-


###scrapy crawl vsco

# from scrapy.cmdline import execute
# print('cmdline//////')
# execute()

import warnings
warnings.filterwarnings('ignore')
import sys
from os import path
sys.path.append( path.dirname(path.dirname(path.abspath(__file__))))


from scrapy.cmdline import execute
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#
# execute(['scrapy', 'crawl', 'vsco',
#          '-a','is_batch=1',###1:yes,0:no
#          '-a', 'input_path=/home/heimi/文档/gitWork/myLearn/test_demo//data/pred_need_trans/small_data.csv',
#          '-a', 'output_path=/home/heimi/文档/gitWork/myLearn/test_demo/data/pred_need_trans/batch_trans_data.csv',
#          ]
#
#         )  # 你需要将此处的spider_name替换为你自己的爬虫名称
#
# # #
#
# execute(['scrapy', 'crawl', 'vsco',
#          '-a','is_batch=0',  ###1:yes,0:no
#          '-a', 'input_path=/home/heimi/文档/gitMeorient/myLearn/test_demo/data/pred_need_trans/parel_3c_all_need_trans.csv',
#          '-a', 'output_path=/home/heimi/文档/gitMeorient/myLearn/test_demo/data/pred_need_trans/parel_3c_all_trans.csv',
#          ]
#
#         )  # 你需要将此处的spider_name替换为你自己的爬虫名称

#
# execute(['scrapy', 'crawl', 'vsco',
#          '-a','is_batch=0',  ###1:yes,0:no
#          '-a', 'input_path=/home/heimi/文档/gitMeorient/myLearn/test_demo/data/tagpack/tagpack1_buy_need_trans.csv',
#          '-a', 'output_path=/home/heimi/文档/gitMeorient/myLearn/test_demo/data/tagpack/tagpack1_trans_ret.csv',
#          ]
#
#         )  # 你需要将此处的spider_name替换为你自己的爬虫名称


execute(['scrapy', 'crawl', 'vsco',
         '-a','is_batch=0',  ###1:yes,0:no
         '-a', 'input_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/tagpack/tagpack1_buy_need_trans_T1.csv',
         '-a', 'output_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/tagpack/tagpack1_trans_ret_T1.csv',
         ]

        )  # 你需要将此处的spider_name替换为你自己的爬虫名称


# execute(['scrapy', 'crawl', 'vsco',
#          '-a','is_batch=0',  ###1:yes,0:no
#          '-a', 'input_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/tagpack2/8.8机器打标标签规则trans.csv',
#          '-a', 'output_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/tagpack2/8.8机器打标标签规则trans_ok.csv',
#          ]
#
#         )  # 你需要将此处的spider_name替换为你自己的爬虫名称
