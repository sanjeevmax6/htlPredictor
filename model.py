import numpy as np
from xgboost import XGBClassifier

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer


def model(training_data, testing_data):
    # Using a pre-existing function to split the training and testing into input parameters (X) and output (y)
    X_train, X_test, y_train, y_test = train_test_split(
        training_data, testing_data, test_size=.3)

    # Initializing a binary search tree to facilitate the running of XGBoost algorithm
    bst = XGBClassifier(n_estimators=2, max_depth=2,
                        learning_rate=1, objective='binary:logistic')

    # Performing matrix manipulations for compatibility
    Y_train = np.empty(shape=[0, ], dtype="float32")
    Y_test = np.empty(shape=[0, ], dtype="float32")

    y_train = np.reshape(y_train, (1, len(y_train)))
    y_test = np.reshape(y_test, (1, len(y_test)))

    Y_train = np.reshape(y_train[0], (len(y_train[0]), ))
    Y_test = np.reshape(y_test[0], (len(y_test[0]), ))

    # Inducing a KNN algorithm to fill in the NaN values by an euclidian algorithm for training and testing data
    imputerTrain = KNNImputer(n_neighbors=int(np.floor(
        np.sqrt(X_train.shape[0]))), weights='uniform', metric='nan_euclidean')
    imputerTrain.fit(X_train)
    X_train = imputerTrain.transform(X_train)

    imputerTest = KNNImputer(n_neighbors=int(
        np.floor(np.sqrt(X_test.shape[0]))), weights='uniform', metric='nan_euclidean')
    imputerTest.fit(X_test)
    X_test = imputerTest.transform(X_test)

    le = LabelEncoder()
    Y_train = le.fit_transform(Y_train)

    # Parsing the data to the tree and training the model
    bst.fit(X_train, Y_train)

    # Parsing the testing data and using the model to predict the data
    preds = bst.predict(X_test)

    return preds, Y_test
