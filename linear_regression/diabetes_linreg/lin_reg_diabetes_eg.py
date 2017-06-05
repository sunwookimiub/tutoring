import numpy as np

# 1. Load data
data_filename = 'X.csv'
X = np.genfromtxt(data_filename, delimiter=' ')
target_filename = 'y.csv'
y = np.genfromtxt(target_filename, delimiter=' ')

Xtrain = X[:-20]
Xtest = X[-20:]
ytrain = y[:-20]
ytest = y[-20:]

# 2. Normalize
for ii in range(Xtrain.shape[1]):
  maxval = np.max(np.abs(Xtrain[:,ii]))
  Xtrain[:,ii] = np.divide(Xtrain[:,ii], maxval)
  Xtest[:,ii] = np.divide(Xtest[:,ii], maxval)

# 3. Add Column of ones
Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))

# 4. Learn
n,m = Xtrain.shape
weights = np.random.rand(m)
err = float('inf')
tolerance = 10e-4
steps = 0

def Errw(X, y, w):
  Xwmy = np.dot(X, w) - y
  Errw = np.dot(Xwmy.T,Xwmy) 
  return Errw

#errors = []
while(np.absolute(Errw(Xtrain, ytrain, weights) - err) > tolerance):
  err = Errw(Xtrain, ytrain, weights) 
  g = 1/float(n) * np.dot(Xtrain.T, np.dot(Xtrain, weights)-ytrain)
  weights = weights - g
  steps += 1
#  errors.append(err)
print("Number of epochs: %d" % steps)
"""

# 5. Test
yhat = np.dot(Xtest, weights)
error = np.linalg.norm(np.subtract(yhat,ytest))/ytest.shape[0] #l2 error
print error
"""


"""
# Plotting Errors
import matplotlib.pyplot as plt
plt.plot(errors[1:50])
plt.show()
plt.plot(errors[1000:])
plt.show()
"""

"""
# Using sklearn
from sklearn import linear_model

regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)
yhat = regr.predict(Xtest)
error = np.linalg.norm(np.subtract(yhat,ytest))/ytest.shape[0] #l2 error
print error
"""

"""
# On scikit-learn using one feature and plotting
Xtrain = Xtrain[:,np.newaxis,2]
#Xtrain = Xtrain[:,None,2]
Xtest = Xtest[:,2]
Xtest = Xtest.reshape(Xtest.shape[0], 1)
Xtest = Xtest
regr.fit(Xtrain, ytrain)
yhat = regr.predict(Xtest)
error = np.linalg.norm(np.subtract(yhat,ytest))/ytest.shape[0] #l2 error
print error
import matplotlib.pyplot as plt
plt.scatter(Xtest, ytest, color='black')
plt.plot(Xtest, yhat, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
"""
