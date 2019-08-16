# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class VcsogridItem(scrapy.Item):
    batch_result = scrapy.Field()
    fromLang = scrapy.Field()
    toLang=scrapy.Field()
    output_path = scrapy.Field()
    batch_idx = scrapy.Field()
    count_total = scrapy.Field()
    is_batch = scrapy.Field()



