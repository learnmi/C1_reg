import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
# Remember what this line did?
#%matplotlib inline  
import math 

# the library we will use to create the model 
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
from sklearn import metrics 

DATA_FILE = "/Users/sonu/Documents/aiml/assignments/c1_m2/father_son_heights.csv"
DATA_SEP = ','

# read inputs
dataset = pd.read_csv(DATA_FILE, sep=DATA_SEP)
print(dataset.head())

# reshape the data
x = dataset['Father'].values.reshape(-1,1)
y = dataset['Son'].values.reshape(-1,1)

# Split data
data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(x, y, test_size=0.3, random_state=3)

# Scatter plot
plt.scatter(x,y)
plt.xlabel("Father's Height")
plt.ylabel("Son's Height")
plt.show()

# Linear Regression
model  = LinearRegression()
model.fit(data_x_train, data_y_train)

# Print model
print('Predicted coeffient value is: ', str(model.coef_))
print('Predicted intercept value is: ', str(model.intercept_))
print("Accuracy is: %", str(100*model.score(data_x_test,data_y_test)))

# Predicted model
data_y_pred = model.predict(data_x_test)

# line and scatter plot for predicted model
plt.scatter(data_x_test,data_y_pred)
sample = np.linspace(start=min(x), stop=max(x), num = 50)
data_sample_pred = model.predict(sample)
plt.plot(sample, data_sample_pred, color='r')
plt.xlabel("Father's Height")
plt.ylabel("Son's Height")
plt.show()


