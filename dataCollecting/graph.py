from alpha_vantage.timeseries import TimeSeries
import matplotlib.pyplot as plt


ts = TimeSeries(key='ZR07VE377CJYBGLJ',output_format='pandas')
comp = 'MSFT'
# Get json object with the intraday data and another with the call's metadata


try:
	print("getting Data for ", comp, "...")
	data, meta_data = ts.get_intraday(symbol=comp,interval='1min', outputsize='full')
	print("Data retrieved for ", comp)
	try:
		data['4. close'].plot()
		plt.title('Intraday Times Series for the MSFT stock (1 min)')
		plt.show()
	except:
		print('data plotting error')
except Exception as e: 
    print("!!ERROR!! ", e)
