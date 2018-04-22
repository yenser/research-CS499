import os
import csv
import libs.bcolors as c

def calcDeltas(increment, stock):
    with open('data/csv/' + stock + '/' + increment + '_' + stock + '.csv', 'r') as csvfile:
        if not os.path.exists('data/delta/' + stock + '/'):
            os.makedirs('data/delta/' + stock + '/')
        with open('data/delta/' + stock + '/' + increment + '_' +stock + '_DELTA.csv', 'w+') as deltafile:
            try:
                readcsv = csv.reader(csvfile, delimiter = ',')
                writecsv = csv.writer(deltafile, delimiter = ',')
                csvlist = list(readcsv)
                i = 2
                for i in range(2, len(csvlist)):
                    k = i
                    j = k - 1
                    #print(csvlist[1][1])
                    writecsv.writerow([float(csvlist[j][1]) - float(csvlist[i][1])])
                print(increment + ": SUCCESS")
            except IndexError:
                print(increment + ": FAILURE")
