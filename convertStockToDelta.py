import convertToDelta

def convertStockToDelta(stock):
    print('----------' + stock + '----------')
    convertToDelta.calcDeltas('DAILY', stock)
    convertToDelta.calcDeltas('WEEKLY', stock)
    convertToDelta.calcDeltas('MONTHLY', stock)
