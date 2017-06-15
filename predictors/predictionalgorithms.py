import numpy as np
import utilities as utils

class Predictor:
    def __init__(self, params={}):
        self.weights = None
        self.params = {}

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        self.weights = np.random.rand(Xtrain.shape[1])

    def predict(self, Xtest):
        """ Most regressors return a dot product for the prediction """
        ytest = np.dot(Xtest, self.weights)
        return ytest

    def reset(self, params):
        utils.update_dictionary_items(self.params, params) 
        

class BatchGradientDescent(Predictor):

    def __init__(self, params={}):
        self.weights = None
        self.params = {'tolerance': 1.0, 'eta0':1.0} # Default
        self.reset(params)

    def Errw(self, X, y, w):
        Xwmy = np.dot(X, w) - y
        Errw = np.dot(Xwmy.T,Xwmy) 
        return Errw

    def learn(self, Xtrain, ytrain):
        n,m = Xtrain.shape
        self.weights = np.random.rand(m)
        err = float('inf')
        steps = 0
        while(np.absolute(self.Errw(Xtrain, ytrain, self.weights) - err) > self.params['tolerance']):
            err = self.Errw(Xtrain, ytrain, self.weights) 
            g = 1/float(n) * np.dot(Xtrain.T, np.dot(Xtrain, self.weights)-ytrain)
            self.weights = self.weights - self.params['eta0']*g
            steps += 1
        print("Number of epochs: %d" % steps)

class StochasticGradientDescent(Predictor):

    def __init__(self, params={}):
        self.weights = None
        self.params = {'epochs': 100, 'eta0':1.0} # Default
        self.reset(params)

    def Errw(self, X, y, w):
        Xwmy = np.dot(X, w) - y
        Errw = np.dot(Xwmy.T,Xwmy) 
        return Errw

    def learn(self, Xtrain, ytrain):
        n,m = Xtrain.shape
        self.weights = np.random.rand(m)
        err = float('inf')
        steps = 0
        for i in range(self.params['epochs']):
            temp = np.c_[Xtrain, ytrain]
            np.random.shuffle(temp)
            ytrain = temp[:,-1]
            Xtrain = temp[:,:-1]
            for t in range(n):
                g = np.dot(np.dot(Xtrain[t].T, self.weights) - ytrain[t], Xtrain[t])
                eta = self.params['eta0']*((1+t)**(-1))
                self.weights = self.weights - eta*g

class LogisticRegression(Predictor):

    def crossEntropy(self, w, Xtrain, Y):
        wx = np.dot(Xtrain, w)
        lhs = np.dot(Y, np.log(utils.sigmoid(wx)))
        rhs = np.dot((1-Y).T, np.log(1-utils.sigmoid(wx)))
        ll = lhs + rhs
        ll = np.negative(ll)
        return ll

    def learn(self, Xtrain, ytrain):
        maxsteps = 10000
        step = 1
        err = float('inf')
        tolerance = 0.0001
        alpha = 0.01 # static stepsize
        self.weights = np.dot(np.linalg.pinv(np.dot(Xtrain.T, Xtrain)), np.dot(Xtrain.T, ytrain))
        while ( np.abs(self.crossEntropy(self.weights, Xtrain, ytrain) - err) > tolerance and step < maxsteps):
            w = self.weights
            err = self.crossEntropy(w, Xtrain, ytrain)
            p = utils.sigmoid(np.dot(Xtrain,w))
            grad = -np.dot(Xtrain.T, np.subtract(ytrain,p)) + reg
            grad = grad/float(Xtrain.shape[0])
            w = w - alpha*grad
            step += 1
            self.weights = w
        print ("Number of steps: %d" % step)

    def predict(self, Xtest):
        wtX = np.dot(Xtest, self.weights.T)
        h = utils.sigmoid(wtX)
        output = (h >= 0.5)*1.0
        return output