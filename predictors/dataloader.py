import numpy as np
import scipy.io as sio

def loadcsv(filename):
    dataset = np.genfromtxt(filename,delimiter=',')
    return dataset

def load_sleep():
    dataset = loadcsv('datasets/sleep_data.csv')
    trainset, testset = splitdataset(dataset) #2 features - 1:coffee, 2:sleep
    return trainset, testset

def load_susy():
    """ A physics classification dataset with 8 features """
    # --- Testing on small ---
    filename = 'datasets/susysubset.csv'
    dataset = loadcsv(filename)
    # ------
    trainset, testset = splitdataset(dataset)    
    return trainset,testset

def load_susy_complete():
    """ A physics classification dataset """
    filename = 'datasets/susycomplete.csv'
    dataset = loadcsv(filename)
    trainset, testset = splitdataset(dataset)    
    return trainset,testset

def load_iris():
  """ Iris Flower Dataset: https://en.wikipedia.org/wiki/Iris_flower_data_set """
  X = np.load("datasets/X_iris_subset.npy")
  y = np.load("datasets/y_iris_subset.npy")

  X = X[:, [0, 3]] # sepal length and petal width

  # Normalize
  for ii in range(X.shape[1]):
    maxval = np.max(np.abs(X[:,ii]))
    if maxval > 0:
      X[:,ii] = np.divide(X[:,ii], maxval)

    # standardize (performs better)
    #X[:,ii] = (X[:,ii] - X[:,ii].mean()) / X[:,ii].std()

  X = np.hstack((X,np.ones((X.shape[0],1))))

  return ((X,y),(X,y))

def load_iris_complete():
  """ Iris Flower Dataset: https://en.wikipedia.org/wiki/Iris_flower_data_set """
  X = np.load("datasets/X_iris.npy")
  y = np.load("datasets/y_iris.npy")

  X = X[:, [0, 3]] # sepal length and petal width

  # Normalize
  for ii in range(X.shape[1]):
    maxval = np.max(np.abs(X[:,ii]))
    if maxval > 0:
      X[:,ii] = np.divide(X[:,ii], maxval)

    # standardize (performs better)
    #X[:,ii] = (X[:,ii] - X[:,ii].mean()) / X[:,ii].std()

  X = np.hstack((X,np.ones((X.shape[0],1))))

  return ((X,y),(X,y))

def load_conc():
    X = sio.loadmat("datasets/concentric.mat")['X'] # 2 x 152 matrix
    X = X.T
    y = np.load('datasets/concentric_y.npy')

    return ((X,y),(X,y))
    

def splitdataset(dataset, target=-1):
    s = int(dataset.shape[0]*0.8)
    Xtrain = dataset[:s,:target]
    ytrain = dataset[:s,target]
    Xtest = dataset[s:,:target]
    ytest = dataset[s:,target]

    # Normalize
    for ii in range(Xtrain.shape[1]):
        maxval = np.max(np.abs(Xtrain[:,ii]))
        if maxval > 0:
            Xtrain[:,ii] = np.divide(Xtrain[:,ii], maxval)
            Xtest[:,ii] = np.divide(Xtest[:,ii], maxval)
    
    # Add column of ones
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))
    
    return ((Xtrain,ytrain), (Xtest,ytest))
