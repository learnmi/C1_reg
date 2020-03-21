
import pandas as pd 
import numpy as np 

import matplotlib.pyplot as plt

# the library we will use to create the model 
from sklearn import linear_model
from sklearn.linear_model import Lasso

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
_X = dfTrain['Father'].values.reshape(-1,1)
y = dfTrain['Son'].values.reshape(-1,1)

## Test data
_X_test = dfTest['Father'].values.reshape(-1,1)
y_test = dfTest['Son'].values.reshape(-1,1)


# Degrees to fit
# MAX_ITER     = 100
DEGREE       = 11

poly = PolynomialFeatures(degree=DEGREE)
X = poly.fit_transform(_X)
# transform features
X_test = poly.fit_transform(_X_test)

test_score = []
training_score = []

lps = np.logspace(-5, 1.25, 100)
for lp in lps:
    # Reg model
    lass = Lasso(alpha=lp, max_iter=10000)
    lass.fit(X,y)
    #predict
    y_pred = lass.predict(X)
    y_pred_test = lass.predict(X_test)
    test_score.append((lp, math.sqrt(metrics.mean_squared_error(y_test, y_pred_test))))
    training_score.append((lp, math.sqrt(metrics.mean_squared_error(y, y_pred))))


print("Best Alpha : ", test_score[test_score.index(min(test_score))][0])
print("Best RMSE : ", test_score[test_score.index(min(test_score))][1])

plt.scatter(lps, [i[1] for i in training_score], color='b', label='Training Error')
plt.scatter(lps, [j[1] for j in test_score], color='r', label='Test Error')
plt.xlabel("Alpha")
plt.ylabel("E-RMS")
plt.legend()
plt.figtext(.6, .8, "Best Alpha = %s" % (str(test_score[test_score.index(min(test_score))][0])))
plt.figtext(.6, .7, "RMSE = %s" % (str(test_score[test_score.index(min(test_score))][1])))
plt.show()

