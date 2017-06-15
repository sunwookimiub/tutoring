import numpy as np

def l2(vec):
    return np.linalg.norm(vec)

def l2err(prediction, ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

def geterror(prediction, ytest):
    # Can change this to other error values
    return l2err(prediction,ytest)/ytest.shape[0]

def update_dictionary_items(dict1, dict2):
    """ Replace any common dictionary items in dict1 with the values in dict2 """
    for k in dict2:
        if k in dict1:
            dict1[k] = dict2[k]
