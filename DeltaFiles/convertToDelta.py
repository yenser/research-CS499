import csv

with open('DAILY_AAPL.csv') as csvfile:
    with open('DAILY_AAPL_DELTA.csv', 'w') as deltafile:

        readcsv = csv.reader(csvfile, delimiter = ',')
        writecsv = csv.writer(deltafile, delimiter = ',')
        csvlist = list(readcsv)

        print (len(csvlist))

        i = 2

        for i in range(2, len(csvlist)):
            k = i
            j = k - 1
            #print(csvlist[1][1])
            writecsv.writerow([str(float(csvlist[j][1]) - float(csvlist[i][1]))])
