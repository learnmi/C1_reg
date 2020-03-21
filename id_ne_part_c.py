
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
# Remember what this line did?
#%matplotlib inline  
import math 

# the library we will use to create the model 
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn import metrics


DATA_FILE = "/Users/sonu/Documents/aiml/assignments/c1_Project_C/concrete_data.csv"
DATA_SEP = ','

# read inputs
df = pd.read_csv(DATA_FILE, sep=DATA_SEP)
print('\nData Summary')
print(df.head())

y = df.pop('concrete_compressive_strength')
X = df.copy()


# Split data
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Linear Regression
model  = LinearRegression()
model.fit(data_x_train, data_y_train)

# Predict
y_pred = model.predict(data_x_test)

# Sample parameters
N = y.size
numFeatures = X.size/N

# Model accuracy
MSE  = metrics.mean_squared_error(y_pred, data_y_test)
RMSE = math.sqrt(MSE)
RSE  = math.sqrt((N * (MSE))/N-2)
R2   = metrics.r2_score(data_y_test, y_pred)

# Print model
print('Intercept: \t', str(model.intercept_))
for i in range(len(X.columns)):
    print(X.columns[i], ":\t", model.coef_[i])
print ('\nMSE: ', MSE, '\nRMSE: ', RMSE, '\nRSE:', RSE, '\nR2:', R2)
