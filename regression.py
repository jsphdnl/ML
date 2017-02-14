import pandas as pd
import quandl
from sklearn import preprocessing, cross_validation, svm
import numpy as np
import math
from sklearn.linear_model import LinearRegression


df = quandl.get("WIKI/GOOGL")
df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]
df["HL_PCT"] = (df["Adj. High"] - df["Adj. Close"])/ df["Adj. Close"] * 100
df["OP_PCT"] = (df["Adj. Close"]- df["Adj. Open"]) / df["Adj. Open"] * 100  
df = df[["Adj. Close", "HL_PCT", "OP_PCT", "Adj. Volume"]]
df.fillna(-99999, inplace=True)
forecast_label = "Adj. Close"
forecast_out = int(math.ceil(0.0001*len(df)))

df["label"] = df[forecast_label].shift(-forecast_out)
df.dropna(inplace=True)
#cl = LinearRegression()
cl = svm.SVR()
X = np.array(df.drop(["label"], 1))
Y = np.array(df["label"])

X = preprocessing.scale(X)
df.dropna(inplace=True)
print (len(X), len(Y))

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y, test_size = 0.2)

cl.fit(X_train, Y_train)

accuracy = cl.score(X_test, Y_test)

print(accuracy)
