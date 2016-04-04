#!/bin/bash
scrapy runspider get_poem.py -o ../data/poem-ids-$(date -d "today" +"%Y%m%d").csv

