#!/bin/bash
scrapy runspider get_poem.py -o ../data/poems-$(date -d "today" +"%Y%m%d%H%M").csv

