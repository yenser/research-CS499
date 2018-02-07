from alpha_vantage.timeseries import TimeSeries
import os
import argparse
from pandas.io.json import json_normalize
from requests import Session
import sys
# from ... misc.bcolors import bcolors


session = Session()
comp = ''
apikey = 'ZR07VE377CJYBGLJ'
companyList = ['MSFT', 'GOOGL', 'AMD', 'T', 'AMZN', 'CSCO', 'INTC', 'AAPL', 'NTGR', 'IBM', 'CMCSA', 'ASUUY', 'S', 'ORCL', 'HP', 'VZ', 'ADBE', 'NVDA']

def parse_args():
	parser = argparse.ArgumentParser(description='Pull Data Program')
	parser.add_argument('-c', '--company', type=str, nargs='?', default='', help='Company Acronym')
	return parser.parse_args()

def writeFile(comp, ext, data, length):
	fname = '../data/'+comp+'/'+length+'_'+comp+ext
	manageFile(fname)
	try:
		text_file = open(fname, "w")
		text_file.write(data)
		text_file.close()
		return True
	except Exception as e:
		return False

def requestMonthly(comp):
	print("----------------------------")
	print('MONTHLY: ', end='')
	sys.stdout.flush()
	url='https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol='+comp+'&outputsize=full&apikey='+apikey+'&datatype=csv'
	res = session.get(url)
	data = res.text

	if "Invalid API call" not in data: 
		if(writeFile(comp, '.csv', data, 'MONTHLY')):
			print("SUCCESS")
		else:
			print('ERROR')
	else:
		print('ERROR')

def requestWeekly(comp):
	print("----------------------------")
	print('WEEKLY: ', end='')
	sys.stdout.flush()
	url='https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol='+comp+'&outputsize=full&apikey='+apikey+'&datatype=csv'
	res = session.get(url)
	data = res.text

	if "Invalid API call" not in data: 
		if(writeFile(comp, '.csv', data, 'WEEKLY')):
			print("SUCCESS")
		else:
			print('ERROR')
	else:
		print('ERROR')

def requestDaily(comp):
	print("----------------------------")
	print('DAILY: ', end='')
	sys.stdout.flush()
	url='https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+comp+'&apikey='+apikey+'&datatype=csv&outputsize=full'
	res = session.get(url)
	data = res.text

	if "Invalid API call" not in data: 
		if(writeFile(comp, '.csv', data, 'DAILY')):
			print("SUCCESS")
		else:
			print('ERROR')
	else:
		print('ERROR')


def requestHourly(comp):
	print("----------------------------")
	print('HOURLY: ', end='')
	sys.stdout.flush()
	url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol='+comp+'&interval=60min&apikey='+apikey+'&datatype=csv&outputsize=compact'
	res = session.get(url)
	data = res.text
	if "Invalid API call" not in data: 
		if(writeFile(comp, '.csv', data, 'HOURLY')):
			print("SUCCESS")
		else:
			print('ERROR')
	else:
		print('ERROR')

def requestHalfHourly(comp):
	print("----------------------------")
	print('HALF_HOURLY: ', end='')
	sys.stdout.flush()
	url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol='+comp+'&interval=30min&apikey='+apikey+'&datatype=csv&outputsize=compact'
	res = session.get(url)
	data = res.text
	if "Invalid API call" not in data: 
		if(writeFile(comp, '.csv', data, 'HALF_HOURLY')):
			print("SUCCESS")
		else:
			print('ERROR')
	else:
		print('ERROR')

def requestQuarterHourly(comp):
	print("----------------------------")
	print('QUARTER_HOURLY: ', end='')
	sys.stdout.flush()
	url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol='+comp+'&interval=15min&apikey='+apikey+'&datatype=csv&outputsize=compact'
	res = session.get(url)
	data = res.text
	if "Invalid API call" not in data: 
		if(writeFile(comp, '.csv', data, 'QUARTER_HOURLY')):
			print("SUCCESS")
		else:
			print('ERROR')
	else:
		print('ERROR')

def requestMinutely(comp):
	print("----------------------------")
	print('MINUTELY: ', end='')
	sys.stdout.flush()
	url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol='+comp+'&interval=1min&apikey='+apikey+'&datatype=csv&outputsize=compact'
	res = session.get(url)
	data = res.text
	if "Invalid API call" not in data: 
		if(writeFile(comp, '.csv', data, 'MINUTELY')):
			print("SUCCESS")
		else:
			print('ERROR')
	else:
		print('ERROR')



def manageDirectory(comp):
	#Check Directory
	if not os.path.exists('../data/'+comp+'/'):
		os.makedirs('../data/'+comp+'/')

def manageFile(fname):
	#Remove fname.json
	if(os.path.isfile(fname)):
			# print('\nRemoving previous file: ', fname)
		os.remove(fname)

			# print('\nRemoving file ran into an issue: ', e)

def getAll(comp):
	try:
		print('\n',comp)
		manageDirectory(comp)
		requestMonthly(comp)
		requestWeekly(comp)
		requestDaily(comp)
		requestHourly(comp)
		requestHalfHourly(comp)
		requestQuarterHourly(comp)
		requestMinutely(comp)
		
	except Exception as e: 
		print("!!ERROR!! ", e)



# Main statement
if __name__ == '__main__':
	args = parse_args()
	comp = args.company

	if (comp):
		getAll(comp)
	else:
		for comp in companyList:
			getAll(comp)
	
