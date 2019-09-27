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

#
# execute(['scrapy', 'crawl', 'vsco',
#          '-a','is_batch=0',  ###1:yes,0:no
#          '-a', 'input_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/tagpack/tagpack1_buy_need_trans_T1.csv',
#          '-a', 'output_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/tagpack/tagpack1_trans_ret_T1.csv',
#          ]
#
#         )  # 你需要将此处的spider_name替换为你自己的爬虫名称


# execute(['scrapy', 'crawl', 'vsco',
#          '-a','is_batch=0',  ###1:yes,0:no
#          '-a', 'input_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/tagpack2/8.8机器打标标签规则trans.csv',
#          '-a', 'output_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/tagpack2/8.8机器打标标签规则trans_ok.csv',
#          ]
#
#         )  # 你需要将此处的spider_name替换为你自己的爬虫名称

#


#
#
# execute(['scrapy', 'crawl', 'vsco',
#          '-a','is_batch=0',  ###1:yes,0:no
#          '-a', 'input_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/tagpack2/tagpack2_buy_need_trans_T0.csv',
#          '-a', 'output_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/tagpack2/tagpack2_buy_need_transed_T0.csv',
#          ]
#
#         )  # 你需要将此处的spider_name替换为你自己的爬虫名称





#
# execute(['scrapy', 'crawl', 'vsco',
#          '-a','is_batch=0',  ###1:yes,0:no
#          '-a', 'input_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/meorient_data/买家数据翻译_need_transed.csv',
#          '-a', 'output_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/meorient_data/买家数据翻译_transed.csv',
#          ]
#
#         )  # 你需要将此处的spider_name替换为你自己的爬虫名称


#
#
# execute(['scrapy', 'crawl', 'vsco',
#          '-a','is_batch=0',  ###1:yes,0:no
#          '-a', 'input_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/meorient_data/无法映射买家部分-翻译_need_transed.csv',
#          '-a', 'output_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/meorient_data/无法映射买家部分-翻译_transed.csv',
#          ]
#
#         )  # 你需要将此处的spider_name替换为你自己的爬虫名称
#
#
#


#
# execute(['scrapy', 'crawl', 'vsco',
#          '-a','is_batch=0',  ###1:yes,0:no
#          '-a', 'input_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/meorient_data/付费买家数据翻译后打标_need_transed.csv',
#          '-a', 'output_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/meorient_data/付费买家数据翻译后打标_transed.csv',
#          ]
#
#         )  # 你需要将此处的spider_name替换为你自己的爬虫名称


#
# execute(['scrapy', 'crawl', 'vsco',
#          '-a','is_batch=0',  ###1:yes,0:no
#          '-a', 'input_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/meorient_data/巴西自取买家翻译+跑模型.xlsx_need_transed.csv',
#          '-a', 'output_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/meorient_data/巴西自取买家翻译+跑模型.xlsx_transed.csv',
#          ]
#
#         )  # 你需要将此处的spider_name替换为你自己的爬虫名称



# execute(['scrapy', 'crawl', 'vsco',
#          '-a','is_batch=0',  ###1:yes,0:no
#          '-a', 'input_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/meorient_data/预注册买家映射标签20190916)_need_trans.csv',
#          '-a', 'output_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/meorient_data/预注册买家映射标签20190916)_transed.csv',
#          ]
#
#         )  # 你需要将此处的spider_name替换为你自己的爬虫名称



execute(['scrapy', 'crawl', 'vsco',
         '-a','is_batch=0',  ###1:yes,0:no
         '-a', 'input_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/meorient_data/预注册买家提交标签信息表9.27_need_transed.csv',
         '-a', 'output_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/meorient_data/预注册买家提交标签信息表9.27_transed.csv',
         ]

        )  # 你需要将此处的spider_name替换为你自己的爬虫名称



# execute(['scrapy', 'crawl', 'vsco',
#          '-a','is_batch=0',  ###1:yes,0:no
#          '-a', 'input_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/tagpack2/tagpack1_buy_need_trans_T0_new2.csv',
#          '-a', 'output_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/tagpack2/tagpack1_trans_ret_T0_new2.csv',
#          ]
#
#         )  # 你需要将此处的spider_name替换为你自己的爬虫名称



#
# execute(['scrapy', 'crawl', 'vsco',
#          '-a','is_batch=0',  ###1:yes,0:no
#          '-a', 'input_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/tagpack/预注册买家有T级 无标签_T0_transneed.csv',
#          '-a', 'output_path=/home/heimi/文档/gitCodeLessData/myLearn/Meorient/data/tagpack/预注册买家有T级 无标签_T0_transedok.csv',
#          ]
#
#         )  # 你需要将此处的spider_name替换为你自己的爬虫名称

