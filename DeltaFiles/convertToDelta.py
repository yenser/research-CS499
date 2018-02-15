import csv

with open('test.csv') as csvfile:
    readcsv = csv.reader(csvfile, delimiter = ',')

    csvlist = list(readcsv)

    print (len(csvlist))

    i = 2

    for i in range(1, len(csvlist)):

        j = i - 1
        #print(csvlist[1][1])
        print(i-1, (float(csvlist[j][0]) - float(csvlist[i][0])),)
