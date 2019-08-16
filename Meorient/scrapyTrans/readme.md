# vsco.co 图片抓取爬虫

基于scrapy，并使用selenium调用chrome加载动态网页，分析最终生成的HTML得到需要的图片数据

## 准备工作
- 安装selenium
```bash
pip install selenium
```

- 下载安装chromeDriver
selenium操作chrome浏览器需要有chromeDriver驱动协助，[下载chromeDriver驱动包](http://chromedriver.storage.googleapis.com/index.html?path=2.7/)
下载完成后复制到python环境目录下的Scripts文件夹中，windows中的路径是`C:\Python27\Scripts`

- 安装pillow包
scrapy官方文档中推荐使用pillow来生成缩略图，并将图片归一化为JPEG/RGB格式,[pillow](http://pillow.readthedocs.org/en/latest/installation.html)

## 完成相关准备后，开始编写爬虫
- 开启图片管道
在settings.py文件中写入
```python
    ITEM_PIPELINES = {'scrapy.contrib.pipeline.images.ImagesPipeline': 1}
    IMAGES_STORE = 'D:\images'  # 系统中需要存放图片的文件夹路径
```

- 在item中定义image_urls
```python
    image_urls = scrapy.Field()
    images = scrapy.Field()
```

- 抓取图片
在start_request中获取分页信息，分别调用chrome浏览器请求每一页的url，`time.sleep(3)`使程序睡眠3秒是为了等待页面JS全部加载完毕，
最后将抓取到的img标签的链接赋值给item，并交由图片管道处理，再未配置的情况下，默认只下载原图尺寸的图片

## 启动爬虫
```bash
scrapy crawl vsco
```