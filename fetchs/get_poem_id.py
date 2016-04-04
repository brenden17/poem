from scrapy.contrib.spiders import CrawlSpider
import scrapy

class PoemIdItem(scrapy.Item):
    poem_id = scrapy.Field()
    
class ArticleSpider(CrawlSpider):
    name = "article"
    allowed_domains = ["http://trove.nla.gov.au"]

    base_url = 'http://trove.nla.gov.au/newspaper/result?l-publictag=poem&q&s='
    max_range = 1000
    qs = range(0, max_range, 20)
    start_urls = [base_url + str(q) for q in qs]
     
    def parse(self, response):
        raw_links = response.xpath('//*[@id="tnewspapers"]').xpath('//ol/li/dl/dt/a/@href').extract()
        ids = [link.replace('?searchTerm=&searchLimits=l-publictag=poem', '').split('/')[3] for link in raw_links]
        for poem_id in ids:
            item = PoemIdItem()
            item['poem_id'] = poem_id
            yield item

