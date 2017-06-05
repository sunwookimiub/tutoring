import dataloader as dtl
import utilities as utils
import regressionalalgorithms as algs
import sys 
import argparse

def parseArguments():
    parser = argparse.ArgumentParser(description='Linear Regression on Parameters vs. Grades')
#    parser.add_argument('input', type=str, help='Input file name')
    parser.add_argument('--tolerance', type=float, default=1.0, help='Tolerance level for gradient descent convergence (default = 1.0)')
    return parser.parse_args()

def main():
    args = parseArguments()
    trainset, testset = dtl.load_sleep()
    #numruns = 1

    regressionalgs = {'BatchGradientDescent': algs.BatchGradientDescent()}

    for learnername, learner in regressionalgs.iteritems():
#        learner.learn(trainset[0], trainset[1], sys.argv[1]) #maxsteps sysargv2
        learner.learn(trainset[0], trainset[1], args.tolerance)
        predictions = learner.predict(testset[0])
        error = utils.geterror(testset[1], predictions)
        print 'Error for {0}: {1}'.format(learnername, str(error))

if __name__ == '__main__':
    main()
