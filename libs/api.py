import os
import argparse
from pandas.io.json import json_normalize
from requests import Session
import sys

session = Session()
apikey = 'ZR07VE377CJYBGLJ'

HEADER = '\033[37m'
OKBLUE = '\033[94m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'


def writeFile(comp, ext, data, length):
	fname = 'data/csv/'+comp+'/'+length+'_'+comp+ext
	manageFile(fname)
	try:
		text_file = open(fname, "w")
		text_file.write(data)
		text_file.close()
		return True
	except Exception as e:
		return False

def requestMonthly(comp):
	print(HEADER,"----------------------------")
	print('MONTHLY: ', end='')
	sys.stdout.flush()
	url='https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol='+comp+'&outputsize=full&apikey='+apikey+'&datatype=csv'
	res = session.get(url)
	data = res.text

	if "Invalid API call" not in data: 
		if(writeFile(comp, '.csv', data, 'MONTHLY')):
			print(OKGREEN, "SUCCESS")
		else:
			print(FAIL,'ERROR')
	else:
		print(FAIL,'ERROR')

def requestWeekly(comp):
	print(HEADER,"----------------------------")
	print('WEEKLY: ', end='')
	sys.stdout.flush()
	url='https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol='+comp+'&outputsize=full&apikey='+apikey+'&datatype=csv'
	res = session.get(url)
	data = res.text

	if "Invalid API call" not in data: 
		if(writeFile(comp, '.csv', data, 'WEEKLY')):
			print(OKGREEN, "SUCCESS")
		else:
			print(FAIL,'ERROR')
	else:
		print(FAIL,'ERROR')

def requestDaily(comp):
	print(HEADER,"----------------------------")
	print('DAILY: ', end='')
	sys.stdout.flush()
	url='https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+comp+'&apikey='+apikey+'&datatype=csv&outputsize=full'
	res = session.get(url)
	data = res.text

	if "Invalid API call" not in data: 
		if(writeFile(comp, '.csv', data, 'DAILY')):
			print(OKGREEN,"SUCCESS")
		else:
			print(FAIL,'ERROR')
	else:
		print(FAIL,'ERROR')


def requestHourly(comp):
	print(HEADER,"----------------------------")
	print('HOURLY: ', end='')
	sys.stdout.flush()
	url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol='+comp+'&interval=60min&apikey='+apikey+'&datatype=csv&outputsize=compact'
	res = session.get(url)
	data = res.text
	if "Invalid API call" not in data: 
		if(writeFile(comp, '.csv', data, 'HOURLY')):
			print(OKGREEN,"SUCCESS")
		else:
			print(FAIL,'ERROR')
	else:
		print(FAIL,'ERROR')

def requestHalfHourly(comp):
	print(HEADER,"----------------------------")
	print('HALF_HOURLY: ', end='')
	sys.stdout.flush()
	url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol='+comp+'&interval=30min&apikey='+apikey+'&datatype=csv&outputsize=compact'
	res = session.get(url)
	data = res.text
	if "Invalid API call" not in data: 
		if(writeFile(comp, '.csv', data, 'HALF_HOURLY')):
			print(OKGREEN,"SUCCESS")
		else:
			print(FAIL,'ERROR')
	else:
		print(FAIL,'ERROR')

def requestQuarterHourly(comp):
	print(HEADER,"----------------------------")
	print('QUARTER_HOURLY: ', end='')
	sys.stdout.flush()
	url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol='+comp+'&interval=15min&apikey='+apikey+'&datatype=csv&outputsize=compact'
	res = session.get(url)
	data = res.text
	if "Invalid API call" not in data: 
		if(writeFile(comp, '.csv', data, 'QUARTER_HOURLY')):
			print(OKGREEN,"SUCCESS")
		else:
			print(FAIL,'ERROR')
	else:
		print(FAIL,'ERROR')

def requestMinutely(comp):
	print(HEADER,"----------------------------")
	print('MINUTELY: ', end='')
	sys.stdout.flush()
	url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol='+comp+'&interval=1min&apikey='+apikey+'&datatype=csv&outputsize=compact'
	res = session.get(url)
	data = res.text
	if "Invalid API call" not in data: 
		if(writeFile(comp, '.csv', data, 'MINUTELY')):
			print(OKGREEN,"SUCCESS")
		else:
			print(FAIL,'ERROR')
	else:
		print(FAIL,'ERROR')



def manageTypeDirectory():
	#Check if directory for file type is in exsistance
	if not os.path.exists('data/csv/'):
		os.makedirs('data/csv/')

def manageDirectory(comp):
	#Check Directory
	if not os.path.exists('data/csv/'+comp+'/'):
		os.makedirs('data/csv/'+comp+'/')

def manageFile(fname):
	#Remove fname.json
	if(os.path.isfile(fname)):
			# print('\nRemoving previous file: ', fname)
		os.remove(fname)

			# print('\nRemoving file ran into an issue: ', e)