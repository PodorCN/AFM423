# Import Library
from asyncio.windows_events import NULL
from audioop import mul
from xml.dom import INVALID_MODIFICATION_ERR
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import random
from sympy import false
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import pandas_datareader as web
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, confusion_matrix, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from datetime import timedelta
import datetime
import sklearn.metrics as metrics

stocksOfInterest = {}

# Ensure Reproducibility and Readibility
plt.rcParams['figure.figsize'] = (7,4.5)
np.random.seed(423)
random.seed(423)

trainTestRatio = 0.75

class Settings:
    def __init__(self) -> None:
        self.trainTestRatio = 0.75
        self.tickers2 = ["AAPL","TSLA","T","SHOP","TD"]
        self.tickers = [ "AAPL","GOOGL", # Tech
                        "TD","GS", # Finance
                        "TSLA","GM", # Auto
                        "XOM","CVX", # Energy
                        "AMD","NVDA",  
                        "PFE",
                        "MCD","KO",]
        
        self.theChosenOnes = ["AAPL","TSLA"]

        self.attributeOfInterest = ["Open","High","Low","Close","Volume"]
        self.stocksOfInterest = {}
        self.indicators = ["MACD","OBV",'PROC',"Stochastic Oscillator"]
        self.indicatorHorizon = 14
        self.predictorHorizon  = np.arange(1,101,10)
setting = Settings()

multpl_stocks = web.get_data_yahoo(setting.tickers,
start = "2017-01-01",
end = "2022-02-25")


class stock:

    def __init__(self, name) -> None:
        self.name = name
        self.IndicatorHorizon = 14
        self.orig = pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
        self.currentdf = self.orig
        self.smoothed = self.orig
        self.currentStat = "Original"
        self.y = []
        self.X = []
        self.predictModel = NULL

        
    def getSmoothed(self, alpha=0.9):
        edata = self.orig.ewm(alpha=alpha).mean()    
        self.smoothed["Close"] = edata
    
    def toSmooth(self):
        if self.currentStat == "Original":
            self.currentdf = self.smoothed
            self.currentStat = "Smoothed"

    def toOrig(self):
         if self.currentStat == "Smoothed":
            self.currentdf = self.orig
            self.currentStat = "Original"

    def changeTimeHorizon(self,target):
        self.predictHorizon = target

    def getOBV(self,smooth):
        df = self.currentdf
        N = len(df["Close"])
        i = 0
        OBV = np.zeros(N)
        while i < (N-1):
            if (df['Close'].values)[i+1] - (df['Close'].values)[i] > 0:
                OBV[i+1] = OBV[i] + (df['Volume'].values)[i+1]
            if (df['Close'].values)[i+1] - (df['Close'].values)[i] == 0:
                OBV[i+1] = OBV[i]
            if (df['Close'].values)[i+1]- (df['Close'].values)[i] < 0:
                OBV[i+1] = OBV[i] - (df['Volume'].values)[i+1]
            i = i + 1
        if (smooth == False):
            self.orig['OBV'] = OBV
        else:
            self.smoothed['OBV'] = OBV

    def getRSI(self,smooth):
        df = self.currentdf["Close"]
        df = df.squeeze()
        n = len(df)
        x0 = df[:n - 1].values
        x1 = df[1:].values
        change = x1 - x0
        avgGain = []
        avgLoss = []
        loss = 0
        gain = 0
        for i in range(self.IndicatorHorizon):
            if change[i] > 0:
                gain += change[i]
            elif change[i] < 0:
                loss += abs(change[i])
        averageGain = gain / self.IndicatorHorizon
        averageLoss = loss / self.IndicatorHorizon
        avgGain.append(averageGain)
        avgLoss.append(averageLoss)
        for i in range(self.IndicatorHorizon, n - 1):
            if change[i] >= 0:
                avgGain.append((avgGain[-1] * 13 + change[i]) / 14.0)
                avgLoss.append((avgLoss[-1] * 13) / 14.0)
            else:
                avgGain.append((avgGain[-1] * 13) / 14.0)
                avgLoss.append((avgLoss[-1] * 13 + abs(change[i])) / 14.0)
        avgGain = np.array(avgGain)
        avgLoss = np.array(avgLoss)
        RS = avgGain / avgLoss
        RSI = 100 - (100 / (1 + RS))
        RSI = np.append(np.zeros(14),RSI)
        if (smooth == False):
            self.orig["RSI"] = RSI
        else:
            self.smoothed["RSI"] = RSI   

    def getSO(self,smooth):
        df = self.currentdf
        high = df['High']
        low = df['Low']
        close = df['Close']
        N = len(high)
        highestHigh = np.zeros(N)
        highestHigh[0:13] = np.nan
        lowestLow = np.zeros(N)
        lowestLow[0:13] = np.nan
        i = 13
        while i <= (N-1):
            highestHigh[i] = high[i - 13:i + 1].max()
            lowestLow[i] = low[i - 13:i + 1].min()
            i = i + 1

        SO = np.zeros(N)
        SO[0:13] = np.nan
        SO[13:] = 100 * (close[13:] - lowestLow[13:]) / (highestHigh[13:] - lowestLow[13:])
      
        if (smooth == False):
            self.orig['SO'] = SO
        else:
            self.smoothed['SO'] = SO

    def getWilliamsR(self,smooth):
        df = self.currentdf
        high = df['High']
        low = df['Low']
        close = df['Close']
        N = len(high)
        highestHigh = np.zeros(N)
        highestHigh[0:13] = np.nan
        lowestLow = np.zeros(N)
        lowestLow[0:13] = np.nan
        i = 13
        while i <= (N - 1):
            highestHigh[i] = high[i - 13:i + 1].max()
            lowestLow[i] = low[i - 13:i + 1].min()
            i = i + 1

        WR = np.zeros(N)
        WR[0:13] = np.nan
        WR[13:] = -100 * (highestHigh[13:] - close[13:]) / (highestHigh[13:] - lowestLow[13:])
        if (smooth == False):
            self.orig['WilliamsR']= WR
        else:
            self.smoothed['WilliamsR'] = WR

    def getMACD(self,smooth):
        df = self.currentdf
        close = df['Close']
        ma1 = close.ewm(span = 12, min_periods = 12).mean()
        ma2 = close.ewm(span = 26, min_periods = 26).mean()
        macd = ma1 - ma2
        if (smooth == False):
            self.orig['MACD'] = macd
        else:
            self.smoothed['MACD']  = macd
        
    def getPriceRateOfChange(self,smooth):
        df = self.currentdf
        close = df["Close"]
        close = close.squeeze()
        n = len(close)
        x0 = close[:n - setting.indicatorHorizon]
        x1 = close[setting.indicatorHorizon:]
        x0 = np.array(x0)
        x1 = np.array(x1)
        PriceRateOfChange = (x1 - x0) / x0
        PriceRateOfChange = np.append(np.zeros(14),PriceRateOfChange)
        
        if (smooth == False):
            self.orig["PROC"] = PriceRateOfChange
        else:
            self.smoothed["PROC"] = PriceRateOfChange
        
    def getAllIndicators(self,smooth):
        self.getMACD(smooth)
        self.getOBV(smooth)
        self.getPriceRateOfChange(smooth)
        self.getRSI(smooth)
        self.getSO(smooth)
        self.getWilliamsR(smooth)
        self.currentdf=self.orig

    def prepareData(self,predictHorizon,smooth):
        self.X = []
        self.y = []
        if (smooth):
            data = self.smoothed.iloc[24:]
        else:
            data = self.orig.iloc[24:]
        n = len(data["Open"])
        df = data.loc[:,["MACD","OBV","PROC","RSI","SO","WilliamsR"]]
        for i in range(n-predictHorizon):
            self.X.append(df.iloc[i])
            if (data)["Close"][i+predictHorizon] >= (data)["Close"][i]:
                self.y.append(1)
            else:
                self.y.append(0)

AAPLModel = None

def trainAndDisplay(predictHorizon,smooth,chosen,tune,oob=False,oob_estimators=150,ConfusionM=False,AAPL=False,ROC=False,AAPLStock = None):
    returnResult = {}
    
    if (chosen == True):
        sticks = setting.theChosenOnes
    else:
        sticks = setting.tickers
    if AAPL:
        sticks = [AAPLStock]
    oobResult = {}
    
    
    for s in sticks:
        currentStock = stock(s)
        for a in setting.attributeOfInterest:
            currentStock.currentdf[a] = multpl_stocks[a][s]
        stocksOfInterest[s] = currentStock

    for s in sticks:
        currentStock = stock(s)
        for a in setting.attributeOfInterest:
            currentStock.currentdf[a] = multpl_stocks[a][s]
        stocksOfInterest[s] = currentStock

    for t in sticks:
        s = stocksOfInterest[t]
        s.getAllIndicators(smooth)
        s.prepareData(predictHorizon,smooth)
        n = len(s.currentdf["Open"])
        X = s.X[25:]
        y = s.y[25:]
        trainSize = int(np.ceil(n * 0.8))-1
        trainSetX2 = X[:trainSize]
        trainSetY2 = y[:trainSize]
        testSetX = X[trainSize:]
        testSetY = y[trainSize:]
        trainSetX, testSetX, trainSetY, testSetY = train_test_split(trainSetX2, trainSetY2, train_size = (4*trainSize) // 5)
        
        if tune:
            print("Using Grid_Search")
            grid_search = GridSearchCV(RandomForestClassifier(random_state=42,oob_score=True,n_estimators=150),
                               { 'max_features':np.arange(1,6,1),},cv=5, scoring="accuracy",verbose=1,n_jobs=-1
                               )
            
        else:
            print("RandomForestClassifier")
            grid_search = RandomForestClassifier(n_estimators=oob_estimators, random_state=423,oob_score=True)
        
        try:
            grid_search.fit(trainSetX, trainSetY)
        except:
            print(t+" with timeHorizon "+str(predictHorizon))
        if tune:
            print(grid_search.best_params_)
            stocksOfInterest[t].predictModel = grid_search   
        
        pred = grid_search.predict(testSetX)
        precision = precision_score(y_pred=pred, y_true=testSetY)
        recall = recall_score(y_pred=pred, y_true=testSetY)
        f1 = f1_score(y_pred=pred, y_true=testSetY)
        accuracy = accuracy_score(y_pred=pred, y_true=testSetY)
        confusion = confusion_matrix(y_pred=pred, y_true=testSetY)
        
        returnResult[s.name] = accuracy
        
        if (oob):
            oobResult[t] = grid_search.oob_score_
        
        if (ROC):
            
            probs = grid_search.predict_proba(testSetX)
            preds = probs[:,1]
            fpr, tpr, threshold = metrics.roc_curve(testSetY, preds)
            roc_auc = metrics.auc(fpr, tpr)
      
            plt.title('Receiver Operating Characteristic')
            plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
            plt.legend(loc = 'lower right')
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()

    if (oob):
        return(oobResult)
    if (ConfusionM):
        return(confusion)
    return returnResult


## Accuracy VS TimeHorizon
trainResult = []
for i in setting.predictorHorizon:
    print(i)
    trainResult.append(trainAndDisplay(predictHorizon=i,smooth=False,chosen=False,tune=True,oob=False,oob_estimators=150,ConfusionM=False,AAPL=False,ROC=False))
    
plotResultHorizon = {}
for t in setting.tickers:
    tmp = []
    for r in trainResult:
        tmp.append(r[t])
    plotResultHorizon[t] = tmp

plt.figure(figsize=(20,7))
for i in setting.tickers:
    plt.plot(setting.predictorHorizon,plotResultHorizon[i],label = i)
    plt.xlabel("Time Horizon (days)")
    plt.ylabel("Accuracy")
    plt.legend()

## Tuned vs Non-Tuned
for i in setting.theChosenOnes:
    stocksOfInterest[i].toOrig()
    
trainResultNoTuned = []
trainResultTuned = []
for i in setting.predictorHorizon:
    trainResultNoTuned.append(trainAndDisplay(predictHorizon=i,smooth=False,chosen=True,tune=False,oob=False,oob_estimators=150,ConfusionM=False,AAPL=False,ROC=False))
    trainResultTuned.append(trainAndDisplay(predictHorizon=i,smooth=False,chosen=True,tune=True,oob=False,oob_estimators=150,ConfusionM=False,AAPL=False,ROC=False))

plotResultHorizonNoTuned = {}
for t in setting.theChosenOnes:
    tmp = []
    for r in trainResultNoTuned:
        tmp.append(r[t])
    plotResultHorizonNoTuned[t] = tmp
    
    
plotResultHorizonTuned = {}
for t in setting.theChosenOnes:
    tmp = []
    for r in trainResultTuned:
        tmp.append(r[t])
    plotResultHorizonTuned[t] = tmp

plt.figure(figsize=(20,7))
for i in setting.theChosenOnes:
    plt.plot(setting.predictorHorizon,plotResultHorizonNoTuned[i],label = i)
for i in setting.theChosenOnes:
    plt.plot(setting.predictorHorizon,plotResultHorizonTuned[i],label = i+" Tuned")

plt.xlabel("Time Horizon (days)")
plt.ylabel("Accuracy")
plt.legend()

## Smooth vs Non-smooth
for i in setting.theChosenOnes:
    stocksOfInterest[i].getSmoothed()
    stocksOfInterest[i].toSmooth()

    
trainResultSmoothed = []
for i in setting.predictorHorizon:
    trainResultSmoothed.append(trainAndDisplay(predictHorizon=i,smooth=True,chosen=True,tune=False,oob=False,oob_estimators=150,ConfusionM=False,AAPL=False,ROC=False))

plotResultHorizonSmoothed = {}
for t in setting.theChosenOnes:
    tmp = []
    for r in trainResultSmoothed:
        tmp.append(r[t])
    plotResultHorizonSmoothed[t] = tmp
    
for i in setting.theChosenOnes:
    stocksOfInterest[i].toOrig()


## Simulation  
stockInvested = "TD"
Cash = 2000
Cash_Day0 = Cash
cashFlow = []
buySellFlow = []
StockBook = {}

AAPLModel = stocksOfInterest[stockInvested].predictModel
AAPL_Stock = multpl_stocks["Close"][stockInvested]
AAPL_Indicator = stocksOfInterest[stockInvested].orig

newTDays = []
tradingDays = np.array(multpl_stocks["Adj Close"][stockInvested].index)
for i in tradingDays:
    ts = pd.Timestamp(i).strftime("%Y-%m-%d")
    newTDays.append(ts)
    
AAPL_Stock_Buy_Day = np.where(np.array(newTDays) == '2021-10-01')[0][0]
AAPL_Stock_Day0 = AAPL_Stock_Buy_Day
print(AAPL_Stock_Buy_Day)
BuyTDays = []
buyAndHold = 0

# Buy Phase
for i in np.arange(30):
    try:
        todayIndicator = AAPL_Indicator.loc[newTDays[AAPL_Stock_Buy_Day]][["MACD","OBV","PROC","RSI","SO","WilliamsR"]]
    except:
        AAPL_Stock_Buy_Day += 1
        continue
    todayResult = AAPLModel.predict([todayIndicator])
    timeStr = newTDays[AAPL_Stock_Buy_Day]
    if (todayResult == 1):
        # Long
        if (Cash-AAPL_Stock.loc[timeStr] <= 0):
            print("No Money, can't buy on day "+str(AAPL_Stock_Buy_Day))
            AAPL_Stock_Buy_Day += 1
            continue
        StockBook[AAPL_Stock_Buy_Day] = AAPL_Stock.loc[timeStr]
        Cash -= AAPL_Stock.loc[timeStr]
        cashFlow.append(Cash)
        buySellFlow.append(1)
    else:
        # Short
        StockBook[AAPL_Stock_Buy_Day] = -AAPL_Stock.loc[timeStr]
        Cash += AAPL_Stock.loc[timeStr]
        cashFlow.append(Cash)
        buySellFlow.append(0)
    buysell = "Long"
    if (todayResult == 0):
        buysell = "Short"
    print("On Day: "+newTDays[AAPL_Stock_Buy_Day]+" Cash Amount: "+str(Cash)+" We "+buysell)
    BuyTDays.append(AAPL_Stock_Buy_Day)
    AAPL_Stock_Buy_Day += 1
    
# Sell Phase
AAPL_Stock_Sell_Day = AAPL_Stock_Buy_Day+30
AAPL_Stock_Buy_Day = AAPL_Stock_Day0
for i in np.arange(30):
    if AAPL_Stock_Buy_Day not in BuyTDays:
        print("No Buy on day "+ str(AAPL_Stock_Buy_Day))
        AAPL_Stock_Sell_Day += 1
        AAPL_Stock_Buy_Day += 1
        continue
    
    todayPrice = AAPL_Stock[newTDays[AAPL_Stock_Sell_Day]]
    
    
    if (StockBook[AAPL_Stock_Buy_Day]) > 0:
        buysell = "Sell"
        Cash += todayPrice
        profit = todayPrice - StockBook[AAPL_Stock_Buy_Day]
        cashFlow.append(Cash)
    else:
        buysell = "Buy Back"
        Cash -= todayPrice
        profit = -todayPrice-StockBook[AAPL_Stock_Buy_Day]
        cashFlow.append(Cash)
    print("On Day: "+newTDays[AAPL_Stock_Sell_Day]+ " Cash Amount: "+str(Cash)+" We "+buysell+" with profit "+str(profit))
    AAPL_Stock_Sell_Day += 1
    AAPL_Stock_Buy_Day += 1

print("BuyAndHold: "+str((AAPL_Stock.loc[newTDays[AAPL_Stock_Day0+9]]/AAPL_Stock.loc[newTDays[AAPL_Stock_Day0]])-1))
print("Return:"+str((Cash-Cash_Day0)/Cash_Day0) + " in 3 months "+str(4* (Cash-Cash_Day0)/Cash_Day0)+ " in 1 year!")

## OOB Error Rate vs Time Horizon
oobScoreSeries = []

for i in np.arange(1,200,1):
    print("Processing: "+str(i)+" out of 20")
    oobScoreSeries.append(trainAndDisplay(60,smooth = False,chosen = True,tune = False,oob=True,oob_estimators=i,ConfusionM=false))

oobTickers = {}
for i in setting.theChosenOnes:
    oobTickers[i] = []
    
for i in np.arange(199):
    for t in setting.theChosenOnes:
        oobTickers[t].append(1-oobScoreSeries[i][t])

plt.figure(figsize=(20,7))     
plt.xlabel("Number of Estimators")
plt.ylabel("OOB Error Rate")

for i in setting.theChosenOnes:
    plt.plot(np.arange(1,200,1),oobTickers[i],label = i)
plt.legend()  

## Confusion Matrix
def recall(CM):
    TP = CM[0][0]
    FP = CM[1][0]
    FN = CM[0][1]
    TN = CM[1][1]
    print("Precison: "+str(TP/(TP+FP)))
    print("Sensitivity: "+str(TP/(TP+FN)))
    print("Specificity: "+str(TN/(TN+FP)))
    print("Accuracy: "+str((TP+TN)/(TP+FP+TN+FN)))

CM30 = trainAndDisplay(30,smooth=False,chosen=False,tune=False,oob=False,oob_estimators=200,ConfusionM=True,AAPL=True)
CM60 = trainAndDisplay(60,smooth=False,chosen=False,tune=False,oob=False,oob_estimators=200,ConfusionM=True,AAPL=True)
CM90 = trainAndDisplay(90,smooth=False,chosen=False,tune=False,oob=False,oob_estimators=200,ConfusionM=True,AAPL=True)

recall(CM30)
recall(CM60)
recall(CM90)
