import numpy as np


def mae(preds, y_test):
    #Using the mean absolute error algorithm to predict the accuracy of the model 
    mae = ((np.sum(np.abs(preds-y_test)))/(preds.shape[0]))
    
    return mae
