

import scrapy
import time
from vcsoGrid.items import VcsogridItem
import execjs
import pandas as pd
import re
import json
import os

class ImageSpider(scrapy.spiders.Spider):
	"""
	source_text:
	fromLang:
	toLang:
	"""

	name = 'vsco'
	def __init__(self,is_batch, input_path,output_path, *args, **kwargs):
		super(ImageSpider, self).__init__(*args, **kwargs)
		self.jsctx = self.getJsctx()
		self.is_batch=int(is_batch)
		self.input_path=input_path
		self.output_path=output_path
		self.data=pd.read_csv(self.input_path)
		if self.is_batch:
			print('scrapy batch///////')
		else:
			print('scapy single/////////')

		# tolang_list = ['ar', 'pl', 'ru', 'tr', 'es','pt']

	def start_requests(self):
		for v in self.data.groupby(['fromLang','toLang']):
			[fromLang, toLang] = v[0]
			td = v[1]
			td.index = range(len(td))
			batch_idx_list, batch_line_list, batch_str_list = self.to_batch_data(td)
			count_total = len(batch_idx_list)
			if fromLang==toLang:
				print('fromlang ',fromLang,'tolang',toLang,'is the same!!!!!!!!!!!!!!!!!!!!')
				continue

			url='https://translate.google.cn/translate_a/single'
			for k,(idx_lines,lines,content) in enumerate(zip(batch_idx_list, batch_line_list,batch_str_list)):

				print ('request k',k,' content //len/',len(content),'fromlang',fromLang,'tolang',toLang)
				res = ''
				tk = self.jsctx.call("TL", content)
				param = {'tk': tk, 'q': content}
				param = dict(param, **{
					'client': 'webapp'
					, 'sl': fromLang
					, 'tl': toLang
					, 'hl': toLang
					, 'dt': 'at'
					, 'dt': 'bd'
					, 'dt': 'ex'
					, 'dt': 'ld'
					, 'dt': 'md'
					, 'dt': 'qca'
					, 'dt': 'rw'
					, 'dt': 'rm'
					, 'dt': 'ss'
					, 'dt': 't'
					, 'otf': '1'
					, 'ssel': '0'
					, 'tsel': '3'
					, 'kc': '6'
				})
				myrequest=scrapy.FormRequest(url=url,
											 # headers=self.headers,
											 method='get',
											 callback=self.parse,
											 meta={'idx_lines':idx_lines,
												   'lines':lines,
												   'content':content,
												   'batch_idx':k,
												   'fromLang':fromLang,
												   'toLang':toLang,
												   'count_total':count_total,
												   'is_batch':self.is_batch,
												   },
											 formdata=param)

				yield myrequest

	def parse(self, response):
		url=response.url
		idx_lines=response.meta['idx_lines']
		lines=response.meta['lines']
		content=response.meta['content']
		batch_idx=response.meta['batch_idx']
		fromLang=response.meta['fromLang']
		toLang=response.meta['toLang']
		count_total=response.meta['count_total']
		is_batch=response.meta['is_batch']

		result = json.loads(response.text)
		transtext = ''.join([t[0] for t in result[0] if t[0] is not None])
		print('trans text///',transtext)

		transtext2 = re.sub(';', '', transtext)  ##drop ;
		transtext2 = re.sub('(^\s*)|(\s*$)', '', transtext2)  ##drop head and tail space
		transtext2 = re.sub('(^[\s.,]*mytest)', 'mytest', transtext2)
		idx_pattern = 'mytest[0-9]{1,5}@yeah.net'
		transtext2 = re.sub(idx_pattern, '!@#', transtext2)  ##replace index flag to the same flag

		idx_list = [int(re.findall('[0-9]\d*', v.replace(' ', ''))[0]) for v in
					re.findall(idx_pattern, transtext)]  ##match idx
		trans_list = re.sub('(^\!@#)', '', transtext2).split('!@#')
		trans_pd = pd.DataFrame([idx_list, trans_list], index=['idx', 'trans_text']).T
		trans_pd['idx'] = trans_pd['idx'].astype(int)

		line_pd = pd.DataFrame(lines, columns=['source_text'])
		line_pd['idx'] = idx_lines

		merge_pd = pd.merge(line_pd, trans_pd, on=['idx'], how='left')
		merge_pd['batch_idx'] = batch_idx
		merge_pd['fromLang'] = fromLang
		merge_pd['toLang'] = toLang


		item = VcsogridItem()
		item['batch_result']=merge_pd
		item['fromLang']=fromLang
		item['toLang'] = toLang
		item['output_path']=self.output_path
		item['batch_idx']=batch_idx
		item['count_total']=count_total
		item['is_batch']=self.is_batch


		yield item

	def to_batch_data(self,pl_data):
		batch_line_list = []
		batch_idx_list = []
		batch_str_list = []
		line_list = []
		idx_list = []
		str_cat = ''
		for k, v in enumerate(pl_data['source_text'].tolist()):
			str_cat += (('; mytest%s@yeah.net; ' % (k)) + v)
			line_list.append(v)
			idx_list.append(k)
			str_max=(4000 if self.is_batch == 1 else 1)
			if len(str_cat) > str_max:
				batch_str_list.append(str_cat)
				batch_line_list.append(line_list)
				batch_idx_list.append(idx_list)
				###reset batch
				str_cat = ''
				line_list = []
				idx_list = []
		##append last batch
		if len(str_cat)>0:
			batch_str_list.append(str_cat)
			batch_line_list.append(line_list)
			batch_idx_list.append(idx_list)
		return batch_idx_list, batch_line_list, batch_str_list


	def getJsctx(self):
		return execjs.compile(""" 
	            function TL(a) { 
	            var k = ""; 
	            var b = 406644; 
	            var b1 = 3293161072; 

	            var jd = "."; 
	            var $b = "+-a^+6"; 
	            var Zb = "+-3^+b+-f"; 

	            for (var e = [], f = 0, g = 0; g < a.length; g++) { 
	                var m = a.charCodeAt(g); 
	                128 > m ? e[f++] = m : (2048 > m ? e[f++] = m >> 6 | 192 : (55296 == (m & 64512) && g + 1 < a.length && 56320 == (a.charCodeAt(g + 1) & 64512) ? (m = 65536 + ((m & 1023) << 10) + (a.charCodeAt(++g) & 1023), 
	                e[f++] = m >> 18 | 240, 
	                e[f++] = m >> 12 & 63 | 128) : e[f++] = m >> 12 | 224, 
	                e[f++] = m >> 6 & 63 | 128), 
	                e[f++] = m & 63 | 128) 
	            } 
	            a = b; 
	            for (f = 0; f < e.length; f++) a += e[f], 
	            a = RL(a, $b); 
	            a = RL(a, Zb); 
	            a ^= b1 || 0; 
	            0 > a && (a = (a & 2147483647) + 2147483648); 
	            a %= 1E6; 
	            return a.toString() + jd + (a ^ b) 
	        }; 

	        function RL(a, b) { 
	            var t = "a"; 
	            var Yb = "+"; 
	            for (var c = 0; c < b.length - 2; c += 3) { 
	                var d = b.charAt(c + 2), 
	                d = d >= t ? d.charCodeAt(0) - 87 : Number(d), 
	                d = b.charAt(c + 1) == Yb ? a >>> d: a << d; 
	                a = b.charAt(c) == Yb ? a + d & 4294967295 : a ^ d 
	            } 
	            return a 
	        } 
	        """)

