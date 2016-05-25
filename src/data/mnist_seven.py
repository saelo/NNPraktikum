# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import shuffle
from data.data_set import DataSet


class MNISTSeven(object):
    """
    Small subset (5000 instances) of MNIST data to recognize the digit 7

    Parameters
    ----------
    dataPath : string
        Path to a CSV file with delimiter ',' and unint8 values.
    numTrain : int
        Number of training examples.
    numValid : int
        Number of validation examples.
    numTest : int
        Number of test examples.

    Attributes
    ----------
    trainingSet : list
    validationSet : list
    testSet : list
    """

    # dataPath = "data/mnist_seven.csv"

    def __init__(self, data_path,
                 num_train=3000,
                 num_valid=1000,
                 num_test=1000):

        self.training_set = []
        self.validation_set = []
        self.test_set = []

        self.load(data_path, num_train, num_valid, num_test)

    def load(self, data_path, num_train, num_valid, num_test):
        """Load the data."""
        print("Loading data from " + data_path + "...")

        data = np.genfromtxt(data_path, delimiter=",", dtype="uint8")

        # The last numTest instances ALWAYS comprise the test set.
        train, test = data[:num_train+num_valid], data[num_train+num_valid:]
        shuffle(train)

        train, valid = train[:num_train], train[num_train:]

        self.training_set = DataSet(train)
        self.validation_set = DataSet(valid)
        self.test_set = DataSet(test)

        print("Data loaded.")
