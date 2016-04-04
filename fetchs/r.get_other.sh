#!/bin/bash
scrapy runspider get_other.py -o ../data/others-$(date -d "today" +"%Y%m%d%H%M").csv

