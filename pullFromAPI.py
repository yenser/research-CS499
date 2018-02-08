from alpha_vantage.timeseries import TimeSeries
import os
import argparse
from pandas.io.json import json_normalize
from requests import Session
import sys
from libs import api

comp = ''
companyList = ['MSFT', 'GOOGL', 'AMD', 'T', 'AMZN', 'CSCO', 'INTC', 'AAPL', 'NTGR', 'IBM', 'CMCSA', 'ASUUY', 'S', 'ORCL', 'HP', 'VZ', 'ADBE', 'NVDA']


HEADER = '\033[37m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

def parse_args():
	parser = argparse.ArgumentParser(description='Pull Data Program')
	parser.add_argument('-c', '--company', type=str, nargs='?', default='', help='Company Acronym')
	return parser.parse_args()


def getAll(comp):
	try:
		print('\n',HEADER,comp)
		api.manageTypeDirectory()
		api.manageDirectory(comp)
		api.requestMonthly(comp)
		api.requestWeekly(comp)
		api.requestDaily(comp)
		# api.requestHourly(comp)
		# api.requestHalfHourly(comp)
		# api.requestQuarterHourly(comp)
		# api.requestMinutely(comp)
		
	except Exception as e: 
		print(FAIL,"!!ERROR!! ", e)



# Main statement
if __name__ == '__main__':
	args = parse_args()
	comp = args.company

	if (comp):
		getAll(comp)
	else:
		for comp in companyList:
			getAll(comp)
	

	print(HEADER)
