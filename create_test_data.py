#!/usr/bin/env python3

import numpy as np
from libs import data
import pickle

# def sample_handling(arr1, arr2, arr3, arr4, arr5):
# 	featureset = []
# 	for i in range(len(arr5)):

# 		val1 = val2 = val3 = val4 = 0

# 		if arr1[i] >= 0:
# 			val1 = 1
# 		if arr2[i] >= 0:
# 			val2 = 1
# 		if arr3[i] >= 0:
# 			val3 = 1
# 		if arr4[i] >= 0:
# 			val4 = 1

# 		if arr5[i] >= 0:
# 			featureset.append([[val1, val2, val3, val4], [1,0]])
# 			# featureset.append([_, [1,0]])
# 		else:
# 			featureset.append([[val1, val2, val3, val4], [0,1]])
# 			# featureset.append([_, [0,1]])
# 	return featureset

def sample_handling(arr1, arr2, arr3, arr4, arr5, arr6):
	featureset = []
	for i in range(len(arr6)):

		val1 = arr1[i]
		val2 = arr2[i]
		val3 = arr3[i]
		val4 = arr4[i]
		val5 = arr5[i]
		val6 = arr6[i]

		if arr6[i] >= 0:
			featureset.append([[val1, val2, val3, val4, val5], [1,0]])
			# featureset.append([_, [1,0]])
		else:
			featureset.append([[val1, val2, val3, val4, val5], [0,1]])
			# featureset.append([_, [0,1]])
	return featureset

def create_test_sets(arr1, arr2, arr3, arr4, arr5, arr6, test_size=0.1):
	features = []

	features += sample_handling(arr1, arr2, arr3, arr4, arr5, arr6)

	testing_size = int(test_size*len(features))

	features = np.array(features, dtype=object)
	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])

	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	return train_x, train_y, test_x, test_y


def get_data_and_create_test_set(str1, str2, str3, str4, str5, str6):
	# get data - set to smallest size - flip array
	AAPL = data.get_file_data(str1)
	MSFT = data.get_file_data(str2)
	GOOGL = data.get_file_data(str3)
	AMZN = data.get_file_data(str4)
	ADBE = data.get_file_data(str5)
	ORCL = data.get_file_data(str6)
	AAPL, MSFT, GOOGL, AMZN, ADBE = data.size_test_results(AAPL, MSFT, GOOGL, AMZN, ADBE, ORCL)
	AAPL = data.flip_array(AAPL)
	MSFT = data.flip_array(MSFT)
	GOOGL = data.flip_array(GOOGL)
	AMZN = data.flip_array(AMZN)
	ADBE = data.flip_array(ADBE)
	ORCL = DATA.flip_array(ORCL)
	# finish pulling data

	batch_size = len(AAPL)-1
	train_x, train_y, test_x, test_y = create_test_sets(AAPL, MSFT, GOOGL, AMZN, ADBE, ORCL)

	return train_x, train_y, test_x, test_y, batch_size


def get_data_for_joel(str1, str2, str3, str4, str5, str6):
	AAPL = data.get_file_data(str1)
	MSFT = data.get_file_data(str2)
	GOOGL = data.get_file_data(str3)
	AMZN = data.get_file_data(str4)
	ADBE = data.get_file_data(str5)
	ORCL = data.get_file_data(str6)
	AAPL, MSFT, GOOGL, AMZN, ADBE = data.size_test_results(AAPL, MSFT, GOOGL, AMZN, ADBE, ORCL)
	AAPL = data.flip_array(AAPL)
	MSFT = data.flip_array(MSFT)
	GOOGL = data.flip_array(GOOGL)
	AMZN = data.flip_array(AMZN)
	ADBE = data.flip_array(ADBE)
	ORCL = DATA.flip_array(ORCL)

	return AAPL, MSFT, GOOGL, AMZN, ADBE, ORCL

#run code

if __name__ == '__main__':

	# get data - set to smallest size - flip array
	AAPL = data.get_file_data('AAPL')
	MSFT = data.get_file_data('MSFT')
	GOOGL = data.get_file_data('GOOGL')
	AMZN = data.get_file_data('AMZN')
	ADBE = data.get_file_data('ADBE')
	AAPL, MSFT, GOOGL, AMZN, ADBE = data.size_test_results(AAPL, MSFT, GOOGL, AMZN, ADBE, ORCL)
	AAPL = data.flip_array(AAPL)
	MSFT = data.flip_array(MSFT)
	GOOGL = data.flip_array(GOOGL)
	AMZN = data.flip_array(AMZN)
	ADBE = data.flip_array(ADBE)
	# finish pulling data

	train_x, train_y, test_x, test_y = create_test_sets(AAPL, MSFT, GOOGL, AMZN, ADBE, ORCL)

	print('Writing file...')
	with open('dataset.pickle', 'wb') as f:
		pickle.dump([train_x, train_y, test_x, test_y], f)
	print('Writing success!')

