import pandas as pd
import numpy as np
from dict import *

def importData(dataset_path):
    
    #Reading from dataset using pandas
    dataset = pd.read_excel(dataset_path, index_col=0)

    #Replacing the null values defined by '-' to NaN(definite null value representation in python)
    dataset = dataset.replace('-', np.nan)

    #Converting the column "Catalyst" to a boolean represntation (0 or 1) from a string representation
    dataset['Catalyst'] = dataset['Catalyst'].map(catalyst)

    # Performing one hot encoding for the column 'Reactor Type'
    dataset = pd.get_dummies(dataset, columns=['Reactor Type'])

    # dataset['Reactor Type'] = dataset['Reactor Type'].map(reactor_type)
    
    #Splitting the dataset into training and testing
    testing_data = dataset.loc[:, ['Oil Yield']]
    training_data = dataset.drop("Oil Yield", axis="columns")

    return training_data, testing_data
