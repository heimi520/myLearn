#coding:utf8
import execjs
import requests
import logging
#import cx_Oracle
import json, time
from urllib.parse import urlencode
import pandas as pd
import re


COOKIE="NID=186=Qt78iteoiOwZDCIAK7F7BwF8JZjVsIE2ELGzbSLIxwZ1Cg_li7Z0S497gzec4yW67n9wcv2MCBobq3GC3EoS0CHx_0s0c3oYgsp4tG_vPdQSOmBDAmKXvqxbVPrbzlsaJRKMX8hZXOJ9nv5SDwz3SLUDxTKl_00DnTMUylwC20E; _ga=GA1.3.16817302.1561084149; 1P_JAR=2019-6-21-2"

'''翻译文本
=================

'''
logging.basicConfig(level = logging.INFO, format= '%(asctime)s %(levelname)s %(message)s', datefmt= '%Y-%m-%d %H:%M:%S')




def to_batch_data(pl_data):
    batch_line_list=[]
    batch_idx_list=[]
    batch_str_list=[]
    line_list=[]
    idx_list=[]
    str_cat=''
    for k,v in enumerate(pl_data['PRODUCT_NAME'].tolist()):
        str_cat+=(('; mytest%s@yeah.net; '%(k))+v)
        line_list.append(v)
        idx_list.append(k)
        if len(str_cat)>4300:
            batch_str_list.append(str_cat)
            batch_line_list.append(line_list)
            batch_idx_list.append(idx_list)
            ###reset batch
            str_cat=''
            line_list=[]  
            idx_list=[]
    ##append last batch
    batch_str_list.append(str_cat)
    batch_line_list.append(line_list) 
    batch_idx_list.append(idx_list)
    return batch_idx_list, batch_line_list,batch_str_list

   
    
def trans_batch(js,fromlang,tolang,batch_idx_list, batch_line_list,batch_str_list,batch_trans_list):
#    batch_trans_list=[]
    t0=time.time()
    for k,(idx_lines,lines,str_line) in enumerate(zip(batch_idx_list, batch_line_list,batch_str_list)):
        t1=time.time()
        
#        def trans_sub_batch(js,idx_lines,lines,str_line,fromlang,tolang):
        transtext=js.translate(str_line,fromLang = fromlang, toLang = tolang)
        print('k////',k,'fromlang',fromlang,'tolang//',tolang,'trans len//',len(transtext))
#        break
        transtext2=re.sub(';','',transtext)  ##drop ;
        transtext2=re.sub('(^\s*)|(\s*$)','',transtext2)  ##drop head and tail space
        transtext2=re.sub('(^[\s.,]*mytest)','mytest',transtext2)
        idx_pattern='mytest[0-9]{1,5}@yeah.net'
        transtext2=re.sub(idx_pattern,'!@#',transtext2) ##replace index flag to the same flag
    
        idx_list=[int(re.findall('[0-9]\d*',  v.replace(' ',''))[0]) for v in  re.findall(idx_pattern,transtext)] ##match idx
        trans_list=re.sub('(^\!@#)','',transtext2).split('!@#')                                          
        trans_pd=pd.DataFrame([idx_list,trans_list],index=['idx','trans_text']).T
        trans_pd['idx']=trans_pd['idx'].astype(int)

        

                     
        idx=pd.DataFrame(idx_list,columns=['a'])

        
        line_pd=pd.DataFrame(lines,columns=['source_text'])
        line_pd['idx']=idx_lines

        
        merge_pd=pd.merge(line_pd,trans_pd,on=['idx'],how='left')
        merge_pd['batch_idx']=k    
        merge_pd['lang_from']=fromlang
        merge_pd['langto']=tolang                            
        batch_trans_list.append(merge_pd)
        t2=time.time()
        print('all batch',len(batch_line_list),'batch',k,'takes time',t2-t1,'total time',t2-t0)

#       
#    batch_trans_pd=pd.concat(batch_trans_list,axis=0)
#    batch_trans_pd['langfrom']=fromlang
#    batch_trans_pd['langto']=tolang
#    return batch_trans_pd






class MyGoogleTransTools(object):
    def __init__(self):
        self.jsctx=self.getJsctx()
        

    def  getJsctx(self):
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
        
    def translate(self,content, tk = None, fromLang = 'auto', toLang = 'en', proxies = None):
        '''用GOOGLE翻译
        ================
        当源字符有换行（\n）时，返回一个也会有换行
        '''
        time.sleep(0.5)
        res=''
        if len(content) > 4891:
            print(u"翻译的长度超过限制！！！")
            return res
        if tk is None:
            tk = self.jsctx.call("TL", content)
        param = {'tk': tk, 'q': content}  
        headers = {
            'accept': '*/*'
            , 'accept-encoding': 'gzip, deflate, br'
            , 'accept-language': 'zh-CN,zh;q=0.9,en;q=0.8'
            , 'cache-control': 'no-cache'
            , 'cookie': COOKIE
            , 'pragma': 'no-cache'
            , 'referer': 'https://translate.google.cn/?source=osdd'
            , 'user-agent': 'Mozilla/5.0 (Linux; Android 4.0.4; Galaxy Nexus Build/IMM76B) AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.133 Mobile Safari/535.19'
        }
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
        c=0
        while True:
            c+=1
            if c>1:
                print('retry times///',c)
                time.sleep(1)
            if c>5:
                break
            try:
                r = requests.get("""https://translate.google.cn/translate_a/single""", \
                                 headers = headers,\
                                 params=param,\
                                 timeout = 10,\
                                 proxies = proxies)
                result = r.json()
                if r.status_code == 200:
                    res= ''.join([t[0] for t in result[0] if t[0] is not None])
                    break
            except Exception as e:
                print('error///',e)
        return res
#




aa=[[['; ', ';', None, None, 0], ['mytest247@yeah.net; ', 'mytest247@yeah.net;', None, None, 0], ['qr kod kablosuz bluetooth okuyucu 2d tarayıcı; ', 'qr code wireless bluetooth reader 2d scanner;', None, None, 3, None, None, None, [[['f2f34108fc8f5cec195d19405608af7c', 'en_tr_2018q4.md']]]], ['mytest248@yeah.net; ', 'mytest248@yeah.net;', None, None, 0], ["Buttonsmith Van Gogh Yıldızlı Gece Tinker Reel Geri Çekilebilir Rozet Makarası - Timsah Klipsli ve Ekstra Uzun 36 inçlik Standart Hizmet Kordonu - ABD'de üretilmiştir; ", 'Buttonsmith Van Gogh Starry Night Tinker Reel Retractable Badge Reel - with Alligator Clip and Extra-Long 36 inch Standard Duty Cord - Made in The USA;', None, None, 3, None, None, None, [[['f2f34108fc8f5cec195d19405608af7c', 'en_tr_2018q4.md']]]], ['mytest249@yeah.net; ', 'mytest249@yeah.net;', None, None, 0], ['Moukey MMs-2 Ayarlanabilir Masa Mic Standı Masaüstü Masa Masa Üstü Kısa Mikrofon Kaymaz Mic Klip ile Mavi Yeti Kartopu Için Standı; ', 'Moukey MMs-2 Adjustable Desk Mic Stand Desktop Tabletop Table Top Short Microphone Stand with Non-Slip Mic Clip For Blue Yeti Snowball;', None, None, 3, None, None, None, [[['f2f34108fc8f5cec195d19405608af7c', 'en_tr_2018q4.md']]]], ['mytest250@yeah.net; ', 'mytest250@yeah.net;', None, None, 0], ['AbaoSisters Çiçek Kız Elbise Fantezi Tül Saten Dantel Cap Kollu Pageant Kızlar Balo Beyaz Fildişi; ', 'AbaoSisters Flower Girl Dress Fancy Tulle Satin Lace Cap Sleeves Pageant Girls Ball Gown White Ivory;', None, None, 3, None, None, None, [[['f2f34108fc8f5cec195d19405608af7c', 'en_tr_2018q4.md']]]], ['mytest251@yeah.net; ', 'mytest251@yeah.net;', None, None, 0], ['3.5 disket sata ssd dizüstü sabit disk taşınabilir dijital harici sabit disk; ', '3.5 floppy disk sata ssd laptop hard disk portable digital external hard drive;', None, None, 3, None, None, None, [[['f2f34108fc8f5cec195d19405608af7c', 'en_tr_2018q4.md']]]], ['mytest252@yeah.net; ', 'mytest252@yeah.net;', None, None, 0], ['Çok Noktalı Ahşap Kapılar için Pilli Kolay Kontrol NFC Akıllı Kilit; ', 'Multi points Easy Control Nfc Smart Lock with Battery for Wooden Doors;', None, None, 3, None, None, None, [[['f2f34108fc8f5cec195d19405608af7c', 'en_tr_2018q4.md']]]], ['mytest253@yeah.net; ', 'mytest253@yeah.net;', None, None, 0], ['yeni tasarım kızlar pamuklu ceket tasarımcısı sahte kürk palto kadın; ', 'new design girls cotton coat designer faux fur overcoat woman;', None, None, 3, None, None, None, [[['f2f34108fc8f5cec195d19405608af7c', 'en_tr_2018q4.md']]]], ['mytest254@yeah.net; ', 'mytest254@yeah.net;', None, None, 0], ['Cadılar bayramı kostümleri kadın dans kostüm seksi yetişkin ucuz kostüm yetişkin parti için, veda partisi için en iyi elbise; ', 'Halloween costumes women dance costume sexy adult cheap costume for adult party, best dress for farewell party;', None, None, 3, None, None, None, [[['f2f34108fc8f5cec195d19405608af7c', 'en_tr_2018q4.md']]]], ['mytest255@yeah.net; ', 'mytest255@yeah.net;', None, None, 0], ['PowerEdge R330 raf tipi sunucu Intel Xeon E3-1270 v6 3.8 GHz 8M önbellek 4C / 8T turbo (72W); ', 'PowerEdge R330 rack server Intel Xeon E3-1270 v6 3.8GHz 8M cache 4C/8T turbo (72W);', None, None, 3, None, None, None, [[['f2f34108fc8f5cec195d19405608af7c', 'en_tr_2018q4.md']]]], ['mytest256@yeah.net; ', 'mytest256@yeah.net;', None, None, 0], ['Ever-Pretty Kadınlar Köpüklü Kademeli Şampanya Altın Pullu Mermaid Cap Kollu Abiye Balo Elbise 08999; ', 'Ever-Pretty Women Sparkling Gradual Champagne Gold Sequin Mermaid Cap Sleeves Evening Dress Prom Dress 08999;', None, None, 3, None, None, None, [[['f2f34108fc8f5cec195d19405608af7c', 'en_tr_2018q4.md']]]], ['mytest257@yeah.net; ', 'mytest257@yeah.net;', None, None, 0], ['2019 Kadınlar Termal Spor Giyim İç Çamaşırı Kadın Paçalı Don; ', '2019 Women Thermal Sports Wear Underwear Women Long Johns;', None, None, 3, None, None, None, [[['f2f34108fc8f5cec195d19405608af7c', 'en_tr_2018q4.md']]]], ['mytest258@yeah.net; ', 'mytest258@yeah.net;', None, None, 0], ['4001 1CH kablosuz çağrı cihazı, kablosuz çağrı sunucusu; ', '4001 1CH wireless pager, wireless calling server;', None, None, 3, None, None, None, [[['f2f34108fc8f5cec195d19405608af7c', 'en_tr_2018q4.md']]]], ['mytest259@yeah.net; ', 'mytest259@yeah.net;', None, None, 0], ['Sıcak satış tur masası dizüstü soğutma pedi mobil dizüstü standı; ', 'Hot sale lap desk laptop cooling pad mobile laptop stand;', None, None, 3, None, None, None, [[['f2f34108fc8f5cec195d19405608af7c', 'en_tr_2018q4.md']]]], ['mytest260@yeah.net; ', 'mytest260@yeah.net;', None, None, 0], ['Toplu beyaz kredi kartı USB disk 1 gb 2 gb 4 gb 8 gb 16 gb 32 gb 64 gb boş kartvizit USB flash sürücü; ', 'Bulk white credit card USB disk 1gb 2gb 4gb 8gb 16gb 32gb 64gb blank business card USB flash drive;', None, None, 3, None, None, None, [[['f2f34108fc8f5cec195d19405608af7c', 'en_tr_2018q4.md']]]], ['mytest261@yeah.net; ', 'mytest261@yeah.net;', None, None, 0], ['FermuarStop Toptan Yetkili Distribütörü YKK® 34 "Vislon Fermuar ~ YKK # 5 Kalıp Plastik ~ Ayırma - 580 Siyah (1 Fermuar / Paket); mytest262@yeah.net; HALO Power Püskül Çanta ve Çantalar İçin Taşınabilir Şarj Cihazı ', 'ZipperStop Wholesale Authorized Distributor YKKÂ® 34" Vislon Zipper ~ YKK #5 Molded Plastic ~ Separating - 580 Black (1 Zipper/ Pack); mytest262@yeah.net; HALO Power Tassel Portable Charger for Purses and Handbags - Sleek Power Bank Battery with Built', None, None, 3, None, None, None, [[['f2f34108fc8f5cec195d19405608af7c', 'en_tr_2018q4.md']]]], ['-in Yıldırım ve USB Kabloları - Şık Kompakt Harici Akü Şarj İstasyonu - Siyah; mytest263@yeah.net; Taşınabilir Klasik Radyo; mytest264@yeah.net; Stokta 1200 adet Kadınlar Çiçek Baskı Boho Yoga Pantolon Harem Pantolon Jogging Yapan Pantolon Kadın Tayt Yoga', '-in Lightning and USB Cables - Stylish Compact External Battery Charging Station - Black; mytest263@yeah.net; Portable Classic Radio; mytest264@yeah.net; 1200pcs in Stock Women Floral Print Boho Yoga Pants Harem Pants Jogger Pants Women Leggings Yoga', None, None, 3, None, None, None, [[['f2f34108fc8f5cec195d19405608af7c', 'en_tr_2018q4.md']]]]], None, 'en']

     

#
#
#js=MyGoogleTransTools()
#
#import pandas as pd
#
#small_data=pd.read_excel('../data/lang_trans/tag .xlsx')
#
#line_list=list(set(small_data['T1'].tolist()+small_data['T2'].tolist()+small_data['Product Tag'].tolist()))
#line_se=pd.Series(line_list)
#subdata=pd.DataFrame(line_se[line_se.notnull()],columns=['PRODUCT_NAME'])
#
###########################################################################
#
#subdata.index=range(len(subdata))
#
#ret_list=[]
#for k, str_line in enumerate(subdata['PRODUCT_NAME']):
#    print(k)
#    transtext=js.translate(str_line,fromLang ='en', toLang = 'ar')
#    ret_list.append([str_line,transtext])
#    
#
#ret_pd=pd.DataFrame(ret_list,columns=['source','trans'])
#
#
