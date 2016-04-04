import re
from random import randint

from scrapy.contrib.spiders import CrawlSpider

import scrapy

class ArticleItem(scrapy.Item):
    article_id = scrapy.Field()
    newspaper = scrapy.Field()
    date = scrapy.Field()
    page = scrapy.Field()
    title = scrapy.Field()
    content = scrapy.Field()
    data_x = scrapy.Field()
    data_y = scrapy.Field()
    data_h = scrapy.Field()
    data_w = scrapy.Field()

def clean(l):
    elem = l[0]
    elem = re.sub(r'\r\n', ' ', ''.join(elem).strip()) if len(elem)>0 else elem
    elem = re.sub(r'\t', ' ', elem)
    elem = re.sub(r'\n', ' ', elem)
    return re.sub(r'\((.*)\)', '', elem)

def clean_number(l):
    return ','.join(l)

class ArticleSpider(CrawlSpider):
    name = "article"
    allowed_domains = ["http://trove.nla.gov.au"]


    other_ids = [ '{:03d}'.format(randint(100, 500)) for _ in range(0, 200)]
    base_url = 'http://trove.nla.gov.au/newspaper/article/'
    start_urls = [base_url + '13' + other_id + '999' for other_id in other_ids[1:200]]
    
    def parse(self, response):
        newspaper = response.xpath('/html/body/div[1]/div[2]/div/ul/li[3]/a/text()').extract()
        date = response.xpath('/html/body/div[1]/div[2]/div/ul/li[4]/a/text()').extract()
        page = response.xpath('/html/body/div[1]/div[2]/div/ul/li[5]/a/text()').extract()
        title = response.xpath('/html/body/div[1]/div[2]/div/ul/li[6]/a/text()').extract()
        data = response.xpath('/html/body/div[2]/div[2]/div[1]/div[2]/div[2]/div[1]/div/form')
        data_x = data.xpath('//input/@data-x').extract()
        data_y = data.xpath('//input/@data-y').extract()
        data_h = data.xpath('//input/@data-h').extract()
        data_w = data.xpath('//input/@data-w').extract()

        article_id = response.url.split('/')[-1]
        content = response.xpath('/html/body/div[2]/div[2]/div[1]/div[2]/div[2]/div[1]/div/form').css('div.read').xpath('text()').extract()

        item = ArticleItem()
        item['article_id'] = article_id
        item['newspaper'] = clean(newspaper)
        item['date'] = clean(date)
        item['page'] = clean(page)
        item['title'] = clean(title)
        item['content'] = '\n'.join(content)
        item['data_x'] = clean_number(data_x)
        item['data_y'] = clean_number(data_y)
        item['data_h'] = clean_number(data_h)
        item['data_w'] = clean_number(data_w)

        yield item

