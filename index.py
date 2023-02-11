import numpy as np
from dataset import *
from model import *
from evaluate import *
import pickle

for i in range(20):
    dataset_path = "./Ds - pyrolysis.xlsx"

    # Function call to import data from dataset (csv)
    training_data, testing_data = importData(dataset_path)

    # Converting the training and testing data to a more efficient type for manipulation
    training_data = training_data.to_numpy()
    testing_data = testing_data.to_numpy()

    # Parsing the training and testing data to the model (XGBoost)
    preds, Y_test = model(training_data, testing_data)

    # Parsing the predicted datapoints for calculating model accuracy
    meanError = mae(preds, Y_test)

    # Printing model accuracy
    print("Model error:", meanError)

    # Writing the output in a text file
    with open('output2.txt', 'a') as f:
        f.write("Iteration")
        f.write(str(i))
        f.write("\n")
        f.write("predictions")
        f.write("\n")
        f.write(str(preds))
        f.write("\n")
        f.write("Actual")
        f.write("\n")
        f.write(str(Y_test))
        f.write("\n")
        f.write("Error")
        f.write("\n")
        f.write(str(meanError))
        f.write("\n")
        f.write("\n")
        f.write("\n")
