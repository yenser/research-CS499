from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt


ts = TimeSeries(key='ZR07VE377CJYBGLJ',output_format='pandas')
comp = 'MSFT'
# Get json object with the intraday data and another with the call's metadata


try:
	print("getting Data for ", comp, "...")
	data, meta_data = ts.get_intraday(symbol=comp,interval='60min', outputsize='full')
	print("Data retrieved for ", comp)
	try:
		data.plot()
		plt.title('Intraday Times Series for the '+comp+' stock (60 min)')
		plt.show()
	except:
		print('data plotting error')
except:
	print("Servers are busy... try again in a bit")




