import dataloader as dtl
import utilities as utils
import regressionalalgorithms as algs

if __name__ == '__main__':

    trainset, testset = dtl.load_sleep()

    #numruns = 1

    """
    # sklearn
    from sklearn import linear_model

    regr = linear_model.LinearRegression()
    regr.fit(trainset[0], trainset[1])
    yhat = regr.predict(testset[0])
    error = np.linalg.norm(np.subtract(yhat,testset[1]))/testset[1].shape[0] #l2 error
    print error
    """

    regressionalgs = {'BatchGradientDescent': algs.BatchGradientDescent()}

    for learnername, learner in regressionalgs.iteritems():
        learner.learn(trainset[0], trainset[1], 1) #Do sys argv1
        predictions = learner.predict(testset[0])
        error = utils.geterror(testset[1], predictions)
        print 'Error for {0}: {1}'.format(learnername, str(error))
