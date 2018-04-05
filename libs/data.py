
from numpy import genfromtxt

def readCSV(fileName):
	data = genfromtxt(fileName, delimiter=',')
	return data

def size_test_results(arr1, arr2, arr3, arr4, arr5, arr6):
	smallest = len(arr1)

	if(len(arr2) < smallest):
		smallest = len(arr2)
	if(len(arr3)< smallest):
		smallest = len(arr3)
	if(len(arr4) < smallest):
		smallest = len(arr4)
	if(len(arr5) < smallest):
		smallest = len(arr5)
	if(len(arr6) < smallest):
		smallest = len(arr6)


	return arr1[0:smallest], arr2[0:smallest], arr3[0:smallest], arr4[0:smallest], arr5[0:smallest], arr6[0:smallest]


def get_file_data(comp):
	arr = readCSV('dataNew/delta/'+comp+'/DAILY_'+comp+'_DELTA.csv') # read delta AAPL

	return arr


def flip_array(arr):
	return arr[::-1]
