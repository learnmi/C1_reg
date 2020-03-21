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

class Batcher:
    def __init__(self,x,y):
        self.data_x = x
        self.data_y = y
    
    def totalSamples(self):
        return self.data_x.size
    
    def _getrows(self, numrows):
        if self.totalSamples() > 0:
            if numrows[-1] > self.totalSamples():
                return (self.data_x[numrows[0]:], self.data_y[numrows[0]:])
            else:
                return (self.data_x[numrows], self.data_y[numrows])
        else:
            return None

    def BatchIterator(self, chunksize):
        if self.totalSamples() <= 0:
            return None
        # Provide chunks one by one
        chunkstartmarker = 0
        while chunkstartmarker < self.totalSamples():
            chunkrows = range(chunkstartmarker,chunkstartmarker+chunksize)
            X_chunk, y_chunk = self._getrows(chunkrows)
            yield X_chunk, y_chunk
            chunkstartmarker += chunksize
 
def main():

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

    BATCH_SIZE  = 32
    NUM_EPOCS   = 100

    RANDOM_STATE    = 5
    TOL             = 1e-3
    MAX_ITER        = 50
    ETA0            = .0001

    # Split data
    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)

    for epoc in range(NUM_EPOCS):
        # Create a batcher instance
        batcher = Batcher(data_x_train,data_y_train.ravel())
        batcherator = batcher.BatchIterator(chunksize=BATCH_SIZE)
        # model    
        clf = linear_model.SGDRegressor(tol=TOL, max_iter=MAX_ITER, eta0=ETA0)
        i = 0
        # Train model
        for X_chunk, y_chunk in batcherator:
            clf.partial_fit(X_chunk, y_chunk)
            i = i + 1
        print("Mini batches modelled : ", i)    
        # Now make predictions with trained model
        y_prediction = clf.predict(data_x_test)
        N = data_x_test.size
        # MSE
        MSE  = metrics.mean_squared_error(data_y_test, y_prediction)
        # R2
        R2 = metrics.r2_score(data_y_test, y_prediction)
        # RSE
        RSE  = math.sqrt(N * MSE / (N-2))
        # RMSE
        RMSE = math.sqrt(MSE)
        # Coef
        coef = clf.coef_
        intercept = clf.intercept_

        if R2 > 0.0:
            # Scores
            print ('RSE: ', RSE, '\nMSE:', MSE, '\nRMSE:', RMSE)
            print ('Intercept:', intercept, 'Coef:', coef)
            print ('Score (R2):', clf.score(data_x_test,data_y_test))
            print ('EPOCS:', epoc)
            print ('\n')

            # Plots for Predicted model
            # line and scatter plot for predicted model
            plt.scatter(X,y)
            data_y_pred = clf.predict(data_x_test)
            plt.plot(data_x_test, data_y_pred, color='r')
            plt.xlabel("Father's Height")
            plt.ylabel("Son's Height")
            plt.show()
            continue


main()