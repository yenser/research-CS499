from alpha_vantage.timeseries import TimeSeries
import os
import argparse
from pandas.io.json import json_normalize


comp = ''
fnameJSON = ''
fnamePANDAS = ''
fnameCSV = ''


def parse_args():
	parser = argparse.ArgumentParser(description='Pull Data Program')

	parser.add_argument('company', type=str, nargs=1, help='Company Acronym')

	return parser.parse_args()

def manageFiles(comp):
	#Check Directory
	if not os.path.exists('../data/'+comp+'/'):
		os.makedirs('../data/'+comp+'/')

	#Remove fname.json
	if(os.path.isfile(fnameJSON)):
		try:
			print('Removing previous file: ', fnameJSON)
			os.remove(fnameJSON)
		except Exception as e:
			print('Removing file ran into an issue: ', e)

	#Remove fname.txt (Pandas info)
	if(os.path.isfile(fnamePANDAS)):
		try:
			print('Removing previous file: ', fnameJSON)
			os.remove(fnameJSON)
		except Exception as e:
			print('Removing file ran into an issue: ', e)


# Main statement
if __name__ == '__main__':
	args = parse_args()
	comp = args.company[0]
	fnameJSON = '../data/'+comp+'/'+comp+'.json'
	fnamePANDAS = '../data/'+comp+'/'+comp+'.txt'
	fnameCSV = '../data/'+comp+'/'+comp+'.csv'


	ts = TimeSeries(key='ZR07VE377CJYBGLJ',output_format='json')
	# Get json object with the intraday data and another with the call's metadata

	try:
		print("getting Data for ", comp, "...")
		data, meta_data = ts.get_intraday(symbol=comp,interval='60min', outputsize='full')
		print("Data retrieved for ", comp)

		manageFiles(comp)


		try:
			print("Writing JSON")
			text_file = open(fnameJSON, "w")
			text_file.write(str(data))
			text_file.close()
		except Exception as e:
			print("Writing to file failed: ", e)


		try:
			dataJSON = json_normalize(data)
			print("Writing PANDAS")
			text_file = open(fnamePANDAS, "w")
			text_file.write(str(dataJSON))
			text_file.close()
		except Exception as e:
			print("Writing to file failed: ", e)


		# try:
		# 	dataJSON = json_normalize(data)
		# 	print("Writing CSV")
		# 	text_file = open(fnamePANDAS, "w")
		# 	text_file.write(str(dataJSON))
		# 	text_file.close()
		# except Exception as e:
		# 	print("Writing to file failed: ", e)

	except Exception as e: 
		print("!!ERROR!! ", e)
