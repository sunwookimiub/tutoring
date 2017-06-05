import numpy as np
import utilities as utils

class Regressor:
    def __init__(self):
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        """ Most regressors return a dot product for the prediction """
        ytest = np.dot(Xtest, self.weights)
        return ytest

class BatchGradientDescent(Regressor):
    def Errw(self, X, y, w):
        Xwmy = np.dot(X, w) - y
        Errw = np.dot(Xwmy.T,Xwmy) 
        return Errw

    def learn(self, Xtrain, ytrain, tolerance):
        n,m = Xtrain.shape
        self.weights = np.random.rand(m)
        err = float('inf')
        steps = 0
        while(np.absolute(self.Errw(Xtrain, ytrain, self.weights) - err) > tolerance):
            err = self.Errw(Xtrain, ytrain, self.weights) 
            g = 1/float(n) * np.dot(Xtrain.T, np.dot(Xtrain, self.weights)-ytrain)
            self.weights = self.weights - g
            steps += 1
        print("Number of epochs: %d" % steps)
