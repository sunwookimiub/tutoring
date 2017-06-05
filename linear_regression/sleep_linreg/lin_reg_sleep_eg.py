import numpy as np
dataset = np.genfromtxt('sleep_data.csv',delimiter=',')
Xcoffee = dataset[:,1]
Xsleep = dataset[:,2]
y = dataset[:,3]

Xtrain = Xsleep[:-2]
Xtest = Xsleep[-2:]
ytrain = y[:-2]
ytest = y[-2:]

# Reshape
Xtrain = Xtrain.reshape(-1,1)
#Xtrain = Xtrain.reshape(Xtrain.shape[0],1)
Xtest = Xtest[:,None]
#Xtest = Xtest[:,np.newaxis]

# sklearn
from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)
yhat = regr.predict(Xtest)
error = np.linalg.norm(np.subtract(yhat,ytest))/ytest.shape[0] #l2 error
print error

# 2. Normalize
maxval = np.max(np.abs(Xtrain))
Xtrain = np.divide(Xtrain, maxval)
Xtest = np.divide(Xtest, maxval)

# 3. Add Column of one(s)
Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))

# 4. Learn
n,m = Xtrain.shape
weights = np.random.rand(m)
err = float('inf')

tolerance = 1
#show sysargv

steps = 0
#maybe sysargv2 for maxsteps

def Errw(X, y, w):
  Xwmy = np.dot(X, w) - y
  Errw = np.dot(Xwmy.T,Xwmy) 
  return Errw

errors = []
while(np.absolute(Errw(Xtrain, ytrain, weights) - err) > tolerance):
  err = Errw(Xtrain, ytrain, weights) 
  g = 1/float(n) * np.dot(Xtrain.T, np.dot(Xtrain, weights)-ytrain)
  weights = weights - g
  steps += 1
  errors.append(err)
print("Number of epochs: %d" % steps)
#print yhat
#print ytest

yhat = np.dot(Xtest, weights)
error = np.linalg.norm(np.subtract(yhat,ytest))/ytest.shape[0] #l2 error
#print np.subtract(yhat,ytest) #l2 error
print error

"""
# Plotting Errors
import matplotlib.pyplot as plt
plt.plot(errors[1:int(len(errors)*0.5)])
plt.show()
plt.plot(errors[int(len(errors)*0.5):])
plt.show()
"""
