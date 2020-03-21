import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
# Remember what this line did?
#%matplotlib inline  
import math 

# the library we will use to create the model 
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split

from sklearn import metrics


DATA_FILE = "/Users/sonu/Documents/aiml/assignments/c1_Project_B/father_son_heights.csv"
DATA_SEP = ','

NUM_EPOCS       = 100
TOL             = 1e-6
MAX_ITER        = 50000
ETA0            = .001

# read inputs
dataset = pd.read_csv(DATA_FILE, sep=DATA_SEP)
print('\nData Summary')
print(dataset.head())
print('\n')

# reshape the data
X = dataset['Father'].values.reshape(-1,1)
y = dataset['Son'].values.reshape(-1,1)

# Split data
# data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# Running experiments (50) times.
for epoc in range(NUM_EPOCS):
    clf = linear_model.SGDRegressor(tol=TOL, max_iter=MAX_ITER, eta0=ETA0)
    clf = clf.fit(X,y.ravel())
    N = X.size

    y_prediction = clf.predict(X)

    # MSE
    MSE  = metrics.mean_squared_error(y, clf.predict(X))
    # R2
    R2 = metrics.r2_score(y, clf.predict(X))
    # RSE
    RSE  = math.sqrt(((clf.predict(X) - y) ** 2).sum()/(X.size-2))
    RSE  = math.sqrt(N * MSE / (N-2))
    # RMSE
    RMSE = math.sqrt(MSE)
    # Coef
    coef = clf.coef_
    intercept = clf.intercept_
    
    # Finding the best model fit for Business case.
    # e.g. R2 > 0.2
    if R2 > 0.2:
        print ('RSE: ', RSE, '\nMSE:', MSE, '\nRMSE:', RMSE)
        print ('Intercept:', intercept, 'Coef:', coef)
        print ('Score (R2):', clf.score(X,y))
        print ('Num Iteration:', clf.n_iter_)
        print ('EPOC:', epoc)
        print ('\n')

        # Plots for Predicted model
        # line and scatter plot for predicted model
        plt.scatter(X,y)
        data_y_pred = clf.predict(X)
        plt.plot(X, data_y_pred, color='r')
        plt.xlabel("Father's Height")
        plt.ylabel("Son's Height")
        plt.show()
        break

# Alternate Estimator,
# clf = linear_model.SGDRegressor(tol=1e-6, max_iter=15000, eta0=0.0000001)