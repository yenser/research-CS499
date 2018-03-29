#!/usr/bin/env python3
from alpha_vantage.timeseries import TimeSeries
import os
import argparse
from pandas.io.json import json_normalize
from requests import Session
import sys
from libs import api
from libs import bcolors as c

comp = ''
companyList = ['MSFT', 'GOOGL', 'AMD', 'T', 'AMZN', 'CSCO', 'INTC', 'AAPL', 'NTGR', 'IBM', 'CMCSA', 'ASUUY', 'S', 'ORCL', 'HP', 'VZ', 'ADBE', 'NVDA']


def parse_args():
	parser = argparse.ArgumentParser(description='Pull Data Program')
	parser.add_argument('-c', '--company', type=str, nargs='?', default='', help='Company Acronym')
	return parser.parse_args()


# Main statement
if __name__ == '__main__':
	args = parse_args()
	comp = args.company

	if (comp):
		api.getAll(comp)
	else:
		for comp in companyList:
			api.getAll(comp)
	

