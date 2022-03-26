# Import Library
from asyncio.windows_events import NULL
from audioop import mul
import matplotlib.pyplot as plt
import numpy as np
import random

from sympy import false
#import pandas_techinal_indicators as ta #https://github.com/Crypto-toolbox/pandas-technical-indicators/blob/master/technical_indicators.py
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import pandas_datareader as web
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, confusion_matrix, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# Ensure Reproducibility and Readibility
plt.rcParams['figure.figsize'] = (7,4.5)
np.random.seed(423)
random.seed(423)

trainTestRatio = 0.75

class Settings:
    def __init__(self) -> None:
        self.trainTestRatio = 0.75
        self.tickers = ["AMD", "ATVI", "BABA", "BIDU", "BILI","CEA","GME","GOOGL","HUYA","NVDA"]
        self.attributeOfInterest = ["Open","High","Low","Close","Volume"]
        self.stocksOfInterest = {}
        self.indicators = ["MACD","OBV",'PROC',"Stochastic Oscillator"]
        self.indicatorHorizon = 14
setting = Settings()

# Stocks of interest
multpl_stocks = web.get_data_yahoo(setting.tickers,
start = "2018-11-01",
end = "2020-03-31")

stocksOfInterest = {}

# Indicator of interest


# Get Data

# 各种calculator先define上

# Prepare Data
#   Apply Data Smoothing
#   Get All indicators

# Train Model

# 展示结果
#   画图
#   Metrics  




# Data Smoothing Processor---------------------------------------------------
# Not Done, need look into it
def get_exp_preprocessing(df, alpha=0.9):
    edata = df.ewm(alpha=alpha).mean()    
    return edata
# Data Smoothing Processor---------------------------------------------------

def hyptertune(estimator, X_train, y_train, param_grid, X_test):
    grid_search = GridSearchCV(estimator = estimator, param_grid = param_grid, njobs = -1, verbose = 2)
    grid_search.fit(X_train, y_train)
    pred = grid_search.predict(X_test)
    return pred

# Data Storage --------------------------------------------------------------
class stock:

    def __init__(self, name) -> None:
        self.name = name
        self.IndicatorHorizon = 14
        self.orig = pd.DataFrame(columns=["Open","High","Low","Close","Volume"])
        self.currentdf = self.orig
        self.smoothed = NULL
        self.currentStat = "Original"
        self.day0 = NULL

    def getSmoothed(self):
        self.smoothed = self.orig
        ts = self.smoothed["Close"].squeeze()
        print(ts)
        fit = SimpleExpSmoothing(ts).fit()
        
        self.smoothed["Close"] = fit.fittedvalues.to_frame()
        print(self.smoothed["Close"] )

    def switch(self):
        if self.currentdf == "Original":
            self.currentdf = self.smoothed
            self.currentStat = "Smoothed"
        else:
            self.currentdf = self.orig
            self.currentStat = "Original"

    def getOBV(self):
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
        self.orig['OBV'] = OBV

    def getRSI(self):
        df = self.orig["Close"]
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
        self.orig["RSI"] = RSI
        


    def getSO(self):
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
        self.orig['SO'] = SO
    
    def getWilliamsR(self):
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
        self.orig['WilliamsR'] = WR

    def getMACD(self):
        df = self.currentdf
        close = df['Close']
        ma1 = close.ewm(span = 12, min_periods = 12).mean()
        ma2 = close.ewm(span = 26, min_periods = 26).mean()
        macd = ma1 - ma2
        self.orig['MACD'] = macd

    def getPriceRateOfChange(self):
        df = self.currentdf
        close = df["Close"]
        close = close.squeeze()
        n = len(close)
        x0 = close[:n - setting.indicatorHorizon]
        x1 = close[setting.indicatorHorizon:]
        x0 = np.array(x0)
        x1 = np.array(x1)
        PriceRateOfChange = (x1 - x0) / x0
        self.orig["PROC"] = PriceRateOfChange

    def getAllIndicators(self):
        #self.getMACD() 报错
        self.getOBV()
        self.getPriceRateOfChange()
        self.getRSI()
        self.getSO()
        self.getWilliamsR()
        self.currentdf=self.orig
        #self.getSmoothed()
        #扔掉14天的


for s in setting.tickers:
    currentStock = stock(s)
    for a in setting.attributeOfInterest:
        currentStock.orig[a] = multpl_stocks[a][s]
    stocksOfInterest[s] = currentStock

(stocksOfInterest["AMD"]).getAllIndicators()
print(stocksOfInterest["AMD"].orig)
s=1




data = prepare_data(saapl, 10)

y = data['pred']

#remove the output from the input
features = [x for x in data.columns if x not in ['gain', 'pred']]
X = data[features]

train_size = 2*len(X) // 3

X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]


print('len X_train', len(X_train))
print('len y_train', len(y_train))
print('len X_test', len(X_test))
print('len y_test', len(y_test))


rf = RandomForestClassifier(n_jobs=-1, n_estimators=65, random_state=42)
rf.fit(X_train, y_train.values.ravel());


pred = rf.predict(X_test)
precision = precision_score(y_pred=pred, y_true=y_test)
recall = recall_score(y_pred=pred, y_true=y_test)
f1 = f1_score(y_pred=pred, y_true=y_test)
accuracy = accuracy_score(y_pred=pred, y_true=y_test)
confusion = confusion_matrix(y_pred=pred, y_true=y_test)
print('precision: {0:1.2f}, recall: {1:1.2f}, f1: {2:1.2f}, accuracy: {3:1.2f}'.format(precision, recall, f1, accuracy))
print('Confusion Matrix')
print(confusion)

plt.figure(figsize=(20,7))
plt.plot(np.arange(len(pred)), pred, label='pred')
plt.plot(np.arange(len(y_test)), y_test, label='real' );
plt.title('Prediction versus reality in the test set')
plt.legend();


plt.figure(figsize=(20,7))
proba = rf.predict_proba(X_test)[:,1]
plt.figure(figsize=(20,7))
plt.plot(np.arange(len(proba)), proba, label='pred_probability')
plt.plot(np.arange(len(y_test)), y_test, label='real' );
plt.title('Prediction probability versus reality in the test set');
plt.legend();
plt.show();


rf = RandomForestClassifier(n_jobs=-1, n_estimators=65, random_state=42)
rf.fit(X_train, y_train.values.ravel());


pred = rf.predict(X_test)
precision = precision_score(y_pred=pred, y_true=y_test)
recall = recall_score(y_pred=pred, y_true=y_test)
f1 = f1_score(y_pred=pred, y_true=y_test)
accuracy = accuracy_score(y_pred=pred, y_true=y_test)
confusion = confusion_matrix(y_pred=pred, y_true=y_test)
print('precision: {0:1.2f}, recall: {1:1.2f}, f1: {2:1.2f}, accuracy: {3:1.2f}'.format(precision, recall, f1, accuracy))
print('Confusion Matrix')
print(confusion)

plt.figure(figsize=(20,7))
plt.plot(np.arange(len(pred)), pred, alpha=0.7, label='pred')
plt.plot(np.arange(len(y_test)), y_test, alpha=0.7, label='real' );
plt.title('Prediction versus reality in the test set - Using Leaked data')
plt.legend();

plt.figure(figsize=(20,7))
proba = rf.predict_proba(X_test)[:,1]
plt.figure(figsize=(20,7))
plt.plot(np.arange(len(proba)), proba, alpha = 0.7, label='pred_probability')
plt.plot(np.arange(len(y_test)), y_test, alpha = 0.7, label='real' );
plt.title('Prediction probability versus reality in the test set - Using Leaked data');
plt.legend();
plt.show();