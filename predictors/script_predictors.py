import dataloader as dtl
import utilities as utils
import predictionalgorithms as algs
import sys 
import argparse

def parseArguments():
    parser = argparse.ArgumentParser(description='Linear Predictor on algorithms: Batch Gradient Descent, Stochastic Gradient Descent')
#    parser.add_argument('input', type=str, help='Input file name')
    parser.add_argument('-t', '--tolerance', type=float, default=1.0, help='Tolerance level for Batch gradient descent convergence (default = 1.0)')
    parser.add_argument('-e', '--eta0', type=float, default=1.0, help='Initial Stepsize for gradient descent convergence (default = 1.0)')
    parser.add_argument('-p', '--epochs', type=int, default=1, help='Number of iterations for Stochastic gradient descent convergence (default = 1.0)')
    return parser.parse_args()

def main():
    args = parseArguments()
#    trainset, testset = dtl.load_sleep()
#    regressionalgs = {'BatchGradientDescent': algs.BatchGradientDescent({}),
#                      'StochasticGradientDescent': algs.StochasticGradientDescent({})}
    trainset, testset = dtl.load_susy()
    regressionalgs = {'LogsiticRegression': algs.LogisticRegression({})}
    params = {'tolerance': args.tolerance, 'eta0': args.eta0, 'epochs': args.epochs}

    for learnername, learner in regressionalgs.iteritems():
        learner.reset(params)
        print "Running {0} on {1}".format(learnername, learner.params)
        learner.learn(trainset[0], trainset[1])
        predictions = learner.predict(testset[0])
        error = utils.geterror_classification(testset[1], predictions)
        print 'Error for {0}: {1}\n'.format(learnername, str(error))

if __name__ == '__main__':
    main()
