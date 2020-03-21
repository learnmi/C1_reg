# Step 1 : Build Null model, M0  (1) model
# Step 2 : Build f1, f2,...fd models; select best model from (D-1) models; select based on training data
#        : Build (f5, f1), (f5, f2).... (f5, fd) model  from (D-2) models; select based on training data
#        : Build (f5, f2, f1), (f5, f2, f3), ... model  from (D-3) models; select based on training data
#        : Build (f5 , f2, f3, f1), ... model           (1) model; select based on training data
# Step 3 : Select out of 1 + D-1 + D-2 ....+ 1 models, best model based on Test data

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
# Remember what this line did?
#%matplotlib inline  
import math 
import statistics

# the library we will use to create the model 
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

class FeatureSelector():
    def __init__(self, features):
        self.features = features

    def FeatureSetIterator(self, featureset):
       for afeature in self.features:
           NewSet = list(featureset)
           # Select features
           if afeature not in featureset:
               NewSet.append(afeature)
               yield NewSet
           else:
                continue

class Modeller():
    # Tuning parameters
    MAX_ITER = 10000
    ETA0     = 1e-3
    TOL      = 1e-8
    
    def __init__(self):
        self.model = LinearRegression()

    def Model(self, featureset, data_X, data_y, data_X_test=None, data_y_test=None):
        N = data_y.size
        # Drop features
        _data_X = data_X.copy()
        flt_data_X = pd.concat([_data_X.pop(x) for x in featureset], 1)

        _data_X_test    = data_X_test.copy() if data_X_test is not None else None
        flt_data_X_test = pd.concat([_data_X_test.pop(x) for x in featureset], 1) if _data_X_test is not None else None

        self.model.fit(flt_data_X, data_y)
        # predict, use test?
        if flt_data_X_test is None:
            y_pred = self.model.predict(flt_data_X)
            r2     = metrics.r2_score(data_y, y_pred)
            _MSE  = metrics.mean_squared_error(y_pred, data_y)
            RSE  = math.sqrt((N * (_MSE))/N-2)
        else:
            y_pred = self.model.predict(flt_data_X_test)
            r2     = metrics.r2_score(data_y_test, y_pred)
            _MSE  = metrics.mean_squared_error(y_pred, data_y_test)
            RSE  = math.sqrt((N * (_MSE))/N-2)
        return r2, RSE

class ModelPerformanceEval():

    def __init__(self):
        self.eval = [] #{[f1,f2], r2val} 

    def AppendModelPerformance(self, featureset, R2, RSE=None):
        self.eval.append({'features': featureset, 'score': R2, 'RSE': RSE})

    def getBestModelFeatures(self):
        maxscore = -1
        featureset = []
        for x in self.eval:
            if x['score'] > maxscore:
                maxscore = x['score']
                featureset = x['features']
                rse        = x['RSE']
        return featureset, maxscore, rse

    def getAll(self):
        return self.eval

    def Print(self):
        for i in self.eval:
            print(i)



DATA_FILE = "/Users/sonu/Documents/aiml/assignments/c1_Project_C/concrete_data.csv"
DATA_SEP = ','
FEATURE_VECTOR = ["cement",
                        "blast_furnace_slag",
                        "fly_ash",
                        "water",
                        "superplasticizer",
                        "coarse_aggregate",
                        "fine_aggregate",
                        "age"]

def main():

    # read inputs
    df = pd.read_csv(DATA_FILE, sep=DATA_SEP)
    # print('\nData Summary')
    # print(df.head())

    y = df.pop('concrete_compressive_strength')
    X = df.copy()

    # Sample parameters
    N = y.size
    numFeatures = X.size/N

    # Modeller
    fsel        = FeatureSelector(FEATURE_VECTOR)
    # Training
    modeller    = Modeller()
    globalModelPerf     = ModelPerformanceEval()

    # Test
    testModeller    = Modeller()
    testModelPerf   = ModelPerformanceEval()

    # Split data
    data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    stepfeatureSet = [] # start with no feature

    # Calculate Zero model
    y_pred_model0 = np.empty(data_y_train.size)
    y_pred_model0.fill(statistics.mean(y))
    m0_r2 = metrics.r2_score(data_y_train, y_pred_model0)
    _MSE_m0  = metrics.mean_squared_error(y_pred_model0, data_y_train)
    RSE_m0   = math.sqrt((N * (_MSE_m0))/data_y_train.size-2)

    print("Model-0 features :", {}, "score:", m0_r2, "RSE:", RSE_m0)
    
    numIter = (numFeatures * (numFeatures + 1)/2)
    i = 0
    while (i < numIter):
        fiterator = fsel.FeatureSetIterator(stepfeatureSet)
        modelPerf = ModelPerformanceEval()
        for NewSet in fiterator:
            # get score on training data
            r2, rse = modeller.Model(NewSet, data_x_train, data_y_train)
            modelPerf.AppendModelPerformance(NewSet, r2, rse)
            i = i + 1
        # Select best model performance
        stepfeatureSet, bestScore, rse = modelPerf.getBestModelFeatures()
        globalModelPerf.AppendModelPerformance(stepfeatureSet, bestScore, rse)
    print("Printing performance on Training data.")
    globalModelPerf.Print()

    # Run R2 on training set
    allGoodSets = globalModelPerf.getAll()
    for _set in allGoodSets:
        features = _set['features']
        # get score on test data
        r2, rse = testModeller.Model(features, data_x_train, data_y_train, data_x_test, data_y_test)
        testModelPerf.AppendModelPerformance(features, r2, rse)
    print("Printing performance on Testing data.")        
    testModelPerf.Print()

main()