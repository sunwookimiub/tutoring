import numpy as np

def l2(vec):
    return np.linalg.norm(vec)

def l2err(prediction, ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

def geterror(prediction, ytest):
    # Can change this to other error values
    return l2err(prediction,ytest)/ytest.shape[0]
