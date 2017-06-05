import numpy as np

def loadcsv(filename):
    dataset = np.genfromtxt(filename,delimiter=',')
    return dataset

def load_sleep():
    dataset = loadcsv('sleep_data.csv')
    trainset, testset = splitdataset(dataset, 2) #1:coffee, 2:sleep
    return trainset, testset

def splitdataset(dataset, feature_id):
    Xtrain = dataset[:-2,feature_id]
    ytrain = dataset[:-2,3]
    Xtest = dataset[-2:,feature_id]
    ytest = dataset[-2:,3]

    # Reshape: Because given incorrect representation of (x,) shape
    Xtrain = Xtrain.reshape(-1,1)
    Xtest = Xtest[:,np.newaxis]

    # Normalize
    maxval = np.max(np.abs(Xtrain))
    Xtrain = np.divide(Xtrain, maxval)
    Xtest = np.divide(Xtest, maxval)
    
    # Add column of ones
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))
    
    return ((Xtrain,ytrain), (Xtest,ytest))
