
import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt

# the library we will use to create the model 
from sklearn import linear_model
from sklearn.linear_model import LinearRegression

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

import math

DATA_FILE       = "/Users/sonu/Documents/aiml/assignments/c1_Project_D/project - part D - training data set.csv"
TEST_DATA_FILE  = "/Users/sonu/Documents/aiml/assignments/c1_Project_D/project - part D - testing data set.csv"

DATA_SEP = ','

# read inputs
dfTrain = pd.read_csv(DATA_FILE, sep=DATA_SEP)
dfTest = pd.read_csv(TEST_DATA_FILE, sep=DATA_SEP)

## Train data
X = dfTrain['Father'].values.reshape(-1,1)
y = dfTrain['Son'].values.reshape(-1,1)

## Test data
X_test = dfTest['Father'].values.reshape(-1,1)
y_test = dfTest['Son'].values.reshape(-1,1)

lin = LinearRegression()

# Degrees to fit
N_DEG   = 11
trainingStat = {}
testingStat = {}

for deg in range(1,N_DEG):
    poly    = PolynomialFeatures(degree=deg)
    X_train  = poly.fit_transform(X)
    lin.fit(X_train, y)

    ## predict - training data
    y_pred = lin.predict(X_train)
    # Accuracy
    rmse_ = math.sqrt(metrics.mean_squared_error(y, y_pred))
    trainingStat[str(deg)] = rmse_

    ## predict - test data
    poly_X_test = poly.fit_transform(X_test)
    y_pred_test = lin.predict(poly_X_test)
    # Accuracy
    rmse__ = math.sqrt(metrics.mean_squared_error(y_test, y_pred_test))
    testingStat[str(deg)] = (rmse__)

# Find the optimal degrees
prevStat = 10000000000 #RMSE is undefined for -1 value
bestDeg = 0
for stat in testingStat.keys():
    if testingStat[stat] < prevStat:
        prevStat = testingStat[stat]
        bestDeg = stat
    else:
        continue 
print("Best degree : ", bestDeg)

# Plot
plt.plot(trainingStat.keys(),[trainingStat[i] for i in trainingStat.keys()], color='blue', label='Training Error')
plt.plot(testingStat.keys(),[testingStat[j] for j in testingStat.keys()], color='red', label='Testing Error')
plt.xlabel("N Degrees")
plt.ylabel("E-RMS")
plt.legend()
plt.figtext(.8, .8, "Best Degree = %s" % (str(bestDeg)))
plt.figtext(.8, .7, "RMSE = %s" % (str(prevStat)))
plt.show()


