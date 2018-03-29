import os
import argparse
from pandas.io.json import json_normalize
from requests import Session
import sys
import libs.bcolors as c

session = Session()
apikey = 'ZR07VE377CJYBGLJ'


def writeFile(comp, ext, data, length):
	fname = 'dataNew/csv/'+comp+'/'+length+'_'+comp+ext
	manageFile(fname)
	try:
		text_file = open(fname, "w")
		text_file.write(data)
		text_file.close()
		return True
	except Exception as e:
		return False

def requestMonthly(comp):
	print(c.HEADER,"----------------------------")
	print('MONTHLY: ', end='')
	sys.stdout.flush()
	url='https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol='+comp+'&outputsize=full&apikey='+apikey+'&datatype=csv'
	res = session.get(url)
	data = res.text

	if "Invalid API call" not in data: 
		if(writeFile(comp, '.csv', data, 'MONTHLY')):
			print(c.OKGREEN, "SUCCESS")
		else:
			print(c.FAIL,'ERROR')
	else:
		print(c.FAIL,'ERROR')

def requestWeekly(comp):
	print(c.HEADER,"----------------------------")
	print('WEEKLY: ', end='')
	sys.stdout.flush()
	url='https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol='+comp+'&outputsize=full&apikey='+apikey+'&datatype=csv'
	res = session.get(url)
	data = res.text

	if "Invalid API call" not in data: 
		if(writeFile(comp, '.csv', data, 'WEEKLY')):
			print(c.OKGREEN, "SUCCESS")
		else:
			print(c.FAIL,'ERROR')
	else:
		print(c.FAIL,'ERROR')

def requestDaily(comp):
	print(c.HEADER,"----------------------------")
	print('DAILY: ', end='')
	sys.stdout.flush()
	url='https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol='+comp+'&apikey='+apikey+'&datatype=csv&outputsize=full'
	res = session.get(url)
	data = res.text

	if "Invalid API call" not in data: 
		if(writeFile(comp, '.csv', data, 'DAILY')):
			print(c.OKGREEN,"SUCCESS")
		else:
			print(c.FAIL,'ERROR')
	else:
		print(c.FAIL,'ERROR')


def requestHourly(comp):
	print(c.HEADER,"----------------------------")
	print('HOURLY: ', end='')
	sys.stdout.flush()
	url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol='+comp+'&interval=60min&apikey='+apikey+'&datatype=csv&outputsize=compact'
	res = session.get(url)
	data = res.text
	if "Invalid API call" not in data: 
		if(writeFile(comp, '.csv', data, 'HOURLY')):
			print(c.OKGREEN,"SUCCESS")
		else:
			print(c.FAIL,'ERROR')
	else:
		print(c.FAIL,'ERROR')

def requestHalfHourly(comp):
	print(c.HEADER,"----------------------------")
	print('HALF_HOURLY: ', end='')
	sys.stdout.flush()
	url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol='+comp+'&interval=30min&apikey='+apikey+'&datatype=csv&outputsize=compact'
	res = session.get(url)
	data = res.text
	if "Invalid API call" not in data: 
		if(writeFile(comp, '.csv', data, 'HALF_HOURLY')):
			print(c.OKGREEN,"SUCCESS")
		else:
			print(c.FAIL,'ERROR')
	else:
		print(c.FAIL,'ERROR')

def requestQuarterHourly(comp):
	print(c.HEADER,"----------------------------")
	print('QUARTER_HOURLY: ', end='')
	sys.stdout.flush()
	url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol='+comp+'&interval=15min&apikey='+apikey+'&datatype=csv&outputsize=compact'
	res = session.get(url)
	data = res.text
	if "Invalid API call" not in data: 
		if(writeFile(comp, '.csv', data, 'QUARTER_HOURLY')):
			print(c.OKGREEN,"SUCCESS")
		else:
			print(c.FAIL,'ERROR')
	else:
		print(c.FAIL,'ERROR')

def requestMinutely(comp):
	print(c.HEADER,"----------------------------")
	print('MINUTELY: ', end='')
	sys.stdout.flush()
	url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol='+comp+'&interval=1min&apikey='+apikey+'&datatype=csv&outputsize=compact'
	res = session.get(url)
	data = res.text
	if "Invalid API call" not in data: 
		if(writeFile(comp, '.csv', data, 'MINUTELY')):
			print(c.c.OKGREEN,"SUCCESS")
		else:
			print(c.c.FAIL,'ERROR')
	else:
		print(c.c.FAIL,'ERROR')



def manageTypeDirectory():
	#Check if directory for file type is in exsistance
	if not os.path.exists('dataNew/csv/'):
		os.makedirs('dataNew/csv/')

def manageDirectory(comp):
	#Check Directory
	if not os.path.exists('dataNew/csv/'+comp+'/'):
		os.makedirs('dataNew/csv/'+comp+'/')

def manageFile(fname):
	#Remove fname.json
	if(os.path.isfile(fname)):
			# print('\nRemoving previous file: ', fname)
		os.remove(fname)

			# print('\nRemoving file ran into an issue: ', e)

def getAll(comp):
	try:
		print('\n',c.HEADER,comp)
		manageTypeDirectory()
		manageDirectory(comp)
		requestMonthly(comp)
		requestWeekly(comp)
		requestDaily(comp)
		# api.requestHourly(comp)
		# api.requestHalfHourly(comp)
		# api.requestQuarterHourly(comp)
		# api.requestMinutely(comp)
		print(c.HEADER,"----------------------------")
		
	except Exception as e: 
		print(c.FAIL,"ERROR: ", e)


