import numpy as np

def l2(vec):
    return np.linalg.norm(vec)

def l2err(prediction, ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

def geterror_regression(prediction, ytest):
    # Can change this to other error values
    return l2err(prediction,ytest)/ytest.shape[0]

def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct/float(len(ytest))) * 100.0

def geterror_classification(prediction, ytest):
    return (100.0-getaccuracy(ytest, prediction))

def update_dictionary_items(dict1, dict2):
    """ Replace any common dictionary items in dict1 with the values in dict2 """
    for k in dict2:
        if k in dict1:
            dict1[k] = dict2[k]

def sigmoid(xvec):
    """ Compute the sigmoid function """
    # Cap -xvec, to avoid overflow
    # Undeflow is okay, since it get set to zero
    xvec[xvec < -100] = -100

    vecsig = 1.0 / (1.0 + np.exp(np.negative(xvec)))
 
    return vecsig

def dsigmoid(xvec):
    """ Gradient of standard sigmoid 1/(1+e^-x) """
    vecsig = sigmoid(xvec)
    return vecsig * (1 - vecsig)

def one_class_softmax(W,X,i,k):
    wk = W[:,k].reshape(W.shape[0],1)
    return np.exp(np.dot(wk.T,X[i]))/np.sum(np.exp(np.dot(W.T,X[i])))

def all_class_softmax(W, X):
    return (np.exp(np.dot(X,W).T)/np.sum(np.exp(np.dot(X,W)),axis=1)).T

def one_hot_encoding(y, n_labels):
    mat = np.zeros((len(y), n_labels))
    for i, val in enumerate(y):
        mat[i, val] = 1
    return mat 

