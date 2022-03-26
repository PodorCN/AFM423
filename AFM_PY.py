# Import Library
from asyncio.windows_events import NULL
import matplotlib.pyplot as plt
import numpy as np
import random

from sympy import false
#import pandas_techinal_indicators as ta #https://github.com/Crypto-toolbox/pandas-technical-indicators/blob/master/technical_indicators.py
import pandas as pd

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

# Read Stock Info
# We select Apple,
aapl = pd.read_csv('AAPL.csv')
del(aapl['Date'])
del(aapl['Adj Close'])
aapl.head()

attributeOfInterest = ["Open","High","Low","Close","Volume"]



tickers = ["AMD", "ATVI", "BABA", "BIDU", "BILI","CEA","GME","GOOGL","HUYA","NVDA"]
multpl_stocks = web.get_data_yahoo(tickers,
start = "2018-11-01",
end = "2020-03-31")

stocksOfInterest = []


indicators = ["MACD","OBV",'PROC',"Stochastic Oscillator"]
stocks = ["APPL",]


# Get Data

# 各种calculator先define上

# Prepare Data
#   Apply Data Smoothing
#   Get All indicators

# Train Model

# 展示结果
#   画图
#   Metrics  

def rate_of_change(df, n):
    """
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    M = df['Close'].diff(n - 1)
    N = df['Close'].shift(n - 1)
    ROC = pd.Series(M / N, name='ROC_' + str(n))
    df = df.join(ROC)
    return df

def on_balance_volume(df, n):
    """Calculate On-Balance Volume for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    i = 0
    OBV = [0]
    while i < df.index[-1]:
        if df.loc[i + 1, 'Close'] - df.loc[i, 'Close'] > 0:
            OBV.append(df.loc[i + 1, 'Volume'])
        if df.loc[i + 1, 'Close'] - df.loc[i, 'Close'] == 0:
            OBV.append(0)
        if df.loc[i + 1, 'Close'] - df.loc[i, 'Close'] < 0:
            OBV.append(-df.loc[i + 1, 'Volume'])
        i = i + 1
    OBV = pd.Series(OBV)
    OBV_ma = pd.Series(OBV.rolling(n, min_periods=n).mean(), name='OBV_' + str(n))
    df = df.join(OBV_ma)
    return df

def stochastic_oscillator_k(df):
    """Calculate stochastic oscillator %K for given data.
    
    :param df: pandas.DataFrame
    :return: pandas.DataFrame
    """
    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name='SO%k')
    df = df.join(SOk)
    return df

def macd(df, n_fast, n_slow):
    """Calculate MACD, MACD Signal and MACD difference
    
    :param df: pandas.DataFrame
    :param n_fast: 
    :param n_slow: 
    :return: pandas.DataFrame
    """
    EMAfast = pd.Series(df['Close'].ewm(span=n_fast, min_periods=n_slow).mean())
    EMAslow = pd.Series(df['Close'].ewm(span=n_slow, min_periods=n_slow).mean())
    MACD = pd.Series(EMAfast - EMAslow, name='MACD_' + str(n_fast) + '_' + str(n_slow))
    MACDsign = pd.Series(MACD.ewm(span=9, min_periods=9).mean(), name='MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name='MACDdiff_' + str(n_fast) + '_' + str(n_slow))
    df = df.join(MACD)
    df = df.join(MACDsign)
    df = df.join(MACDdiff)
    return df

def relative_strength_index(df, n):
    """Calculate Relative Strength Index(RSI) for given data.
    
    :param df: pandas.DataFrame
    :param n: 
    :return: pandas.DataFrame
    """
    i = 0
    UpI = [0]
    DoI = [0]
    while i + 1 <= df.index[-1]:
        UpMove = df.loc[i + 1, 'High'] - df.loc[i, 'High']
        DoMove = df.loc[i, 'Low'] - df.loc[i + 1, 'Low']
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)
    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())
    RSI = pd.Series(PosDI / (PosDI + NegDI), name='RSI_' + str(n))
    df = df.join(RSI)
    return df






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
# Indicator Calculator-------------------------------------------------------

class calculator:
    def __init__(self) -> None:
        pass

class indicatorCalculator1(calculator):
    def __init__(self) -> None:
        super().__init__()


class indicatorCalculator2(calculator):
    def __init__(self) -> None:
        super().__init__()


# Indicator Calculator-------------------------------------------------------

for s in tickers:
    stock()
    stocksOfInterest.


for a in attributeOfInterest:
    for 


# Data Storage --------------------------------------------------------------
class stock:
    def __init__(self, name, data) -> None:
        self.name = name
        n = 14
        self.orig = data
        self.currentdf = self.orig
        self.smoothed = NULL
        self.currentStat = "Original"
        self.indicator1 = NULL
        self.indicator2 = NULL
    
    def switch(self):
        if self.currentdf == "Original":
            self.currentdf = self.smoothed
            self.currentStat = "Smoothed"
        else:
            self.currentdf = self.orig
            self.currentStat = "Original"

    def on_balance_volume(self):
        df = self.currentdf
        n = self.n
        i = 0
        OBV = [0]
        while i < df.index[-1]:
            if df.loc[i + 1, 'Close'] - df.loc[i, 'Close'] > 0:
                OBV.append(df.loc[i + 1, 'Volume'])
            if df.loc[i + 1, 'Close'] - df.loc[i, 'Close'] == 0:
                OBV.append(0)
            if df.loc[i + 1, 'Close'] - df.loc[i, 'Close'] < 0:
                OBV.append(-df.loc[i + 1, 'Volume'])
            i = i + 1
        OBV = pd.Series(OBV)
        OBV_ma = pd.Series(OBV.rolling(n, min_periods=n).mean(), name='OBV_' + str(n))
        df = df.join(OBV_ma)
        return df

    def getRSI(self):
        df = self.currentdf
        df = df.squeeze()
        n = len(df)
        x0 = df[:n - 1]
        x1 = df[1:]
        change = x1 - x0
        avgGain = []
        avgLoss = []
        loss = 0
        gain = 0
        for i in range(14):
            if change[i] > 0:
                gain += change[i]
            elif change[i] < 0:
                loss += abs(change[i])
        averageGain = gain / 14.0
        averageLoss = loss / 14.0
        avgGain.append(averageGain)
        avgLoss.append(averageLoss)
        for i in range(14, n - 1):
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
        return np.c_[RSI, x1[13:]]
    
    def getSO(self):
        df = self.currentdf
        high = df[:, 1].squeeze()
        low = df[:, 2].squeeze()
        close = df[:, 3].squeeze()
        n = len(high)
        highestHigh = []
        lowestLow = []
        for i in range(n - 13):
            highestHigh.append(high[i:i + 14].max())
            lowestLow.append(low[i:i + 14].min())
        highestHigh = np.array(highestHigh)
        lowestLow = np.array(lowestLow)
        k = 100 * ((close[13:] - lowestLow) / (highestHigh - lowestLow))

        return np.c_[k, close[13:]]
    
    def getWilliams(self):
        df = self.currentdf
        high = df[:, 1].squeeze()
        low = df[:, 2].squeeze()
        close = df[:, 3].squeeze()
        n = len(high)
        highestHigh = []
        lowestLow = []
        for i in range(n - 13):
            highestHigh.append(high[i:i + 14].max())
            lowestLow.append(low[i:i + 14].min())
        highestHigh = np.array(highestHigh)
        lowestLow = np.array(lowestLow)
        w = -100 * ((highestHigh - close[13:]) / (highestHigh - lowestLow))
        return np.c_[w, close[13:]]

    
    def getMACD(self):
        df = self.currentdf
        ma1 = ema(close.squeeze(), 12)
        ma2 = ema(close.squeeze(), 26)
        macd = ma1[14:] - ma2
        return np.c_[macd, close[len(close) - len(macd):]]
        

    def getPriceRateOfChange(close, n_days):
        close = close.squeeze()
        n = len(close)
        x0 = close[:n - n_days]
        x1 = close[n_days:]
        PriceRateOfChange = (x1 - x0) / x0
        return np.c_[PriceRateOfChange, x1]


    def getOnBalanceVolume(X):
        close = X[:, 3].squeeze()
        volume = X[:, 4].squeeze()[1:]
        n = len(close)
        x0 = close[:n - 1]
        x1 = close[1:]
        change = x1 - x0
        OBV = []
        prev_OBV = 0

        for i in range(n - 1):
            if change[i] > 0:
                current_OBV = prev_OBV + volume[i]
            elif change[i] < 0:
                current_OBV = prev_OBV - volume[i]
            else:
                current_OBV = prev_OBV
            OBV.append(current_OBV)
            prev_OBV = current_OBV
        OBV = np.array(OBV)
        return np.c_[OBV, x1]
# Data Storage --------------------------------------------------------------


saapl = get_exp_preprocessing(aapl)
saapl.head() #saapl stands for smoothed aapl


#for stock in stocks:




def feature_extraction(data):
    for x in [5, 14, 26, 44, 66]:
        data = ta.relative_strength_index(data, n=x)
        data = ta.stochastic_oscillator_d(data, n=x)
        data = ta.accumulation_distribution(data, n=x)
        data = ta.average_true_range(data, n=x)
        data = ta.momentum(data, n=x)
        data = ta.money_flow_index(data, n=x)
        data = ta.rate_of_change(data, n=x)
        data = ta.on_balance_volume(data, n=x)
        data = ta.commodity_channel_index(data, n=x)
        data = ta.ease_of_movement(data, n=x)
        data = ta.trix(data, n=x)
        data = ta.vortex_indicator(data, n=x)
    
    data['ema50'] = data['Close'] / data['Close'].ewm(50).mean()
    data['ema21'] = data['Close'] / data['Close'].ewm(21).mean()
    data['ema14'] = data['Close'] / data['Close'].ewm(14).mean()
    data['ema5'] = data['Close'] / data['Close'].ewm(5).mean()
        
    #Williams %R is missing
    data = ta.macd(data, n_fast=12, n_slow=26)
    
    del(data['Open'])
    del(data['High'])
    del(data['Low'])
    del(data['Volume'])
    
    return data
   
def compute_prediction_int(df, n):
    pred = (df.shift(-n)['Close'] >= df['Close'])
    pred = pred.iloc[:-n]
    return pred.astype(int)

def prepare_data(df, horizon):
    data = feature_extraction(df).dropna().iloc[:-horizon]
    data['pred'] = compute_prediction_int(data, n=horizon)
    del(data['Close'])
    return data.dropna()




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