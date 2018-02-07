from alpha_vantage.timeseries import TimeSeries
import os
import argparse
from pandas.io.json import json_normalize
from requests import Session


session = Session()
comp = ''
apikey = 'ZR07VE377CJYBGLJ'


def parse_args():
	parser = argparse.ArgumentParser(description='Pull Data Program')
	parser.add_argument('company', type=str, nargs=1, help='Company Acronym')
	return parser.parse_args()

def requestMonth(comp):
	print("getting Data for ", comp, "...")
	url='https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol='+comp+'&outputsize=full&apikey='+apikey+'&datatype=csv'
	res = session.get(url)
	return res.text

def requestWeek(comp):
	print("getting Data for ", comp, "...")
	url='https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY&symbol='+comp+'&outputsize=full&apikey='+apikey+'&datatype=csv'
	res = session.get(url)
	return res.text

def requestDaily(comp):
	#not working yet
	print("getting Data for ", comp, "...")
	url='https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol='+comp+'&outputsize=full&apikey='+apikey+'&datatype=csv'
	res = session.get(url)
	return res.text


def manageDirectory(comp):
	#Check Directory
	if not os.path.exists('../data/'+comp+'/'):
		os.makedirs('../data/'+comp+'/')



def manageFile(fname):
	#Remove fname.json
	if(os.path.isfile(fname)):
		try:
			print('Removing previous file: ', fname)
			os.remove(fname)
		except Exception as e:
			print('Removing file ran into an issue: ', e)




def writeFile(comp, ext, data, length):
	fname = '../data/'+comp+'/'+length+'_'+comp+ext
	manageFile(fname)
	try:
		print("Writing file ", fname)
		text_file = open(fname, "w")
		text_file.write(data)
		text_file.close()
		print('Writing complete')
	except Exception as e:
		print("Writing to file failed: ", e)



# Main statement
if __name__ == '__main__':
	args = parse_args()
	comp = args.company[0]


	ts = TimeSeries(key='ZR07VE377CJYBGLJ',output_format='json')
	# Get json object with the intraday data and another with the call's metadata

	try:
		
		manageDirectory(comp)

		# get monthly
		data = requestMonth(comp)
		writeFile(comp, '.csv', data, 'MONTHLY')
		
		# get weekly
		data = requestWeek(comp)
		writeFile(comp, '.csv', data, 'WEEKLY')

		
	except Exception as e: 
		print("!!ERROR!! ", e)
