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

    def __init__(self, params={}):
        self.weights = None
        self.params = {'tolerance': 0.1, 'eta0':1.0} # Default
        self.reset(params)

    def crossEntropy(self, w, Xtrain, Y):
        wx = np.dot(Xtrain, w)
        lhs = np.dot(Y, np.log(utils.sigmoid(wx)))
        rhs = np.dot((1-Y).T, np.log(1-utils.sigmoid(wx)))
        ll = lhs + rhs
        ll = np.negative(ll)
        return ll

    def learn(self, Xtrain, ytrain):
        n = float(Xtrain.shape[0])
        maxsteps = 10000
        step = 0
        err = float('inf')
        self.weights = np.dot(np.linalg.pinv(np.dot(Xtrain.T, Xtrain)), np.dot(Xtrain.T, ytrain))
        while ( np.abs(self.crossEntropy(self.weights, Xtrain, ytrain) - err) > self.params['tolerance'] and step < maxsteps):
            print self.crossEntropy(self.weights, Xtrain, ytrain) - err
            err = self.crossEntropy(self.weights, Xtrain, ytrain)
            p = utils.sigmoid(np.dot(Xtrain,self.weights))
            grad = 1/n * -np.dot(Xtrain.T, np.subtract(ytrain,p))
            self.weights = self.weights - self.params['eta0']*grad
            step += 1
        print ("Number of steps: %d" % step)

    def predict(self, Xtest):
        wtX = np.dot(Xtest, self.weights.T)
        h = utils.sigmoid(wtX)
        output = (h >= 0.5)*1.0
        return output


class SoftmaxRegression(Predictor):

    def __init__(self, params={}):
        self.weights = None
        self.K = 0
        self.params = {'epochs': 0}
        self.reset(params)

    def learn(self, X, y):
        n,m = X.shape
        self.K = np.max(y)+1
        self.weights = np.random.rand(m, self.K) #num features x num classes

        for i in range(self.params['epochs']):
            softm = utils.all_class_softmax(self.weights, X)
            diff = softm - utils.one_hot_encoding(y, self.K)
            g = np.dot(X.T, diff)/n 
            self.weights -= g

            # print (self.softmax_cost_fn(self.weights,X,y)) # compute cost of the whole epoch
   
    def softmax_cost_fn(self,W,X,y):
        return np.sum(self.cross_entropy(W, X, y))/X.shape[0]

    def cross_entropy(self,W,X,y):
        return np.negative(np.sum(np.log(utils.all_class_softmax(W, X)) * utils.one_hot_encoding(y,self.K), axis=1))

    def predict(self, X):
        n = X.shape[0]
        net = np.dot(X, self.weights)
        softm = utils.all_class_softmax(self.weights, X)
        return softm.argmax(axis=1)


class RBFKernel(Predictor):

    def __init__(self, params={}):
        self.weights = None
        self.params = {'tolerance': 0.0, 'eta0': 0.0, 'scale': 0.0}
        self.reset(params)

    def Errw(self, X, y):
        Xwmy = utils.sigmoid(np.dot(X, self.weights)) - y
        Errw = np.dot(Xwmy.T,Xwmy) 
        return Errw/2.0

    def learn(self, X, y):
        X = self.transform(X)
        n,m = X.shape
        self.weights = np.random.normal(loc=0, scale=self.params['scale'],size=4) # random weights
        tot_error = np.inf
        preverror = 0
        while (np.abs(self.Errw(X,y) - preverror) > self.params['tolerance']):
            preverror = self.Errw(X,y)
            grad = np.dot( X.T, (y - utils.sigmoid(np.dot(self.weights, X.T))) * utils.dsigmoid(np.dot(self.weights, X.T)) )
            self.weights = self.weights + self.params['eta0'] * grad
   
    def rbf_kernel(self, X):
        sigma = 0.5
        K = np.zeros((152,152))
        i = 0
        j = 0
        for xi in X:
            for xj in X:
                K[i][j] = np.exp(-np.power(np.linalg.norm(xi-xj),2) / np.power(sigma, 2))
                j += 1
            j = 0
            i += 1
        return K
        # Or 
        # from scipy.spatial.distance import pdist, squareform
        # np.exp(-np.power(squareform(pdist(X, 'euclidean')),2)/np.power(sigma,2))
        # np.exp(-(squareform(pdist(X, 'sqeuclidean')))/np.power(sigma,2))

    def transform(self, X):
        K = self.rbf_kernel(X)
        w,v = np.linalg.eig(K)
        Z = v[:,:3].real
        new_X = np.zeros((152,4))
        for i in range(152):
            new_X[i] = np.hstack((Z[i],1))
        return new_X
        #ax = plt.subplot(111, projection='3d')
        #colors = ['r', 'g', 'b']
        #ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2], c=colors)
        #plt.show()

    def predict(self, X):
        X = self.transform(X)
        testing = utils.sigmoid(np.dot(self.weights, X.T))
        output = (testing > 0.5) * 1.0
        return output

