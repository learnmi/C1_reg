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

# read inputs
dataset = pd.read_csv(DATA_FILE, sep=DATA_SEP)
print('\nData Summary')
print(dataset.head())
print('\n')

# reshape the data
X = dataset['Father'].values.reshape(-1,1)
y = dataset['Son'].values.reshape(-1,1)

NUM_EPOCS       = 1000

TOL             = 1e-6
MAX_ITER        = 10000
ETA0            = .001

# Split data
# Run 50 experiments
for epoc in range(NUM_EPOCS):
    clf = linear_model.SGDRegressor(tol=TOL, max_iter=MAX_ITER, eta0=ETA0)
    clf = clf.partial_fit(X,y.ravel())
    # predict using the model
    y_pred = clf.predict(X)
    N = X.size
    # MSE
    MSE  = metrics.mean_squared_error(y, y_pred)
    # R2
    R2 = metrics.r2_score(y, y_pred)
    # RSE
    RSE  = math.sqrt(N * MSE / (N-2))
    # RMSE
    RMSE = math.sqrt(MSE)
    # Coef
    coef = clf.coef_
    intercept = clf.intercept_
        
    if R2 > 0.23:
        print ('RSE: ', RSE, '\nMSE:', MSE, '\nRMSE:', RMSE)
        print ('Intercept:', intercept, 'Coef:', coef)
        print ('Score (R2):', R2)
        print ('EPOC:', epoc)
        print ('\n')
        # Plots for Predicted model
        # line and scatter plot for predicted model
        plt.scatter(X,y)
        plt.plot(X, y_pred, color='r')
        plt.xlabel("Father's Height")
        plt.ylabel("Son's Height")
        plt.show()
