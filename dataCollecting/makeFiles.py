from alpha_vantage.timeseries import TimeSeries
import os
import argparse
from pandas.io.json import json_normalize
from requests import Session


comp = ''
apikey = 'ZR07VE377CJYBGLJ'


def parse_args():
	parser = argparse.ArgumentParser(description='Pull Data Program')

	parser.add_argument('company', type=str, nargs=1, help='Company Acronym')

	return parser.parse_args()

def requestMonth(comp):

	url='https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY&symbol='+comp+'&interval=15min&outputsize=compact&apikey='+apikey
    res = session.get(url)

    return res.status_code == 200 and 'Login Attempt Failed' not in res.text

def manageDirectory(comp):
	#Check Directory
	if not os.path.exists('../data/'+comp+'/'):
		os.makedirs('../data/'+comp+'/')

def manageFile(comp, ext):
	#Remove fname.json
	fname = comp+ext
	if(os.path.isfile(fname)):
		try:
			print('Removing previous file: ', fname)
			os.remove(fname)
		except Exception as e:
			print('Removing file ran into an issue: ', e)

def writeFile(comp, ext):

	manageFile(comp, ext)

	try:
		text_file = open(fnamePANDAS, "w")
		text_file.write(str(data))
		text_file.close()
	except Exception as e:
		print("Writing to file failed: ", e)



# Main statement
if __name__ == '__main__':
	args = parse_args()
	comp = args.company[0]


	ts = TimeSeries(key='ZR07VE377CJYBGLJ',output_format='json')
	# Get json object with the intraday data and another with the call's metadata

	try:
		print("getting Data for ", comp, "...")
		data, meta_data = ts.get_intraday(symbol=comp,interval='15min', outputsize='full')
		print("Data retrieved for ", comp)

		manageDirectory(comp)
		writeFile(comp, '.json')

		
	except Exception as e: 
		print("!!ERROR!! ", e)
