# -*- coding: utf-8 -*-
import sys
import copy
import logging
import numpy as np
from model.logistic_layer import LogisticLayer
from model.auto_encoder import AutoEncoder
from util.loss_functions import MeanSquaredError

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class DenoisingAutoEncoder(AutoEncoder):
    """
    A denoising autoencoder.
    """

    def __init__(self, train, valid, test, learning_rate=0.1, epochs=30):
        """
         Parameters
        ----------
        train : list
        valid : list
        test : list
        learning_rate : float
        epochs : positive int

        Attributes
        ----------
        training_set : list
        validation_set : list
        test_set : list
        learning_rate : float
        epochs : positive int
        performances: array of floats
        """
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.training_set = copy.deepcopy(train)        # make a copy since we're changing things
        self.validation_set = copy.deepcopy(valid)
        self.test_set = copy.deepcopy(test)

        self.error = MeanSquaredError()

        self.layers = []

        # First hidden layer
        number_of_1st_hidden_layer = 100

        self.layers.append(LogisticLayer(train.input.shape[1],
                                         number_of_1st_hidden_layer, None,
                                         activation="sigmoid",
                                         is_classifier_layer=False))

        # Output layer
        self.layers.append(LogisticLayer(number_of_1st_hidden_layer,
                                         train.input.shape[1], None,
                                         activation="sigmoid",
                                         is_classifier_layer=False))

        # The label is the original input in case of an autoencoder
        self.training_set.labels = self.training_set.input
        # TODO other sets

        # add bias values ("1"s) at the beginning of all data sets
        self.training_set.input = np.insert(self.training_set.input, 0, 1, axis=1)
        self.validation_set.input = np.insert(self.validation_set.input, 0, 1, axis=1)
        self.test_set.input = np.insert(self.test_set.input, 0, 1, axis=1)


    def train(self, verbose=True):
        """
        Train the denoising autoencoder
        """
        # Run the training "epochs" times, print out the logs
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            self._train_one_epoch()

            if verbose:
                # record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                # self.performances.append(accuracy)
                print("loss avg.: {0:.3f}".format(self._err))
                print("-----------------------------")

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer
        """

        # Feed forward layer by layer
        # The output of previous layer is the input of the next layer
        last_layer_output = inp

        for layer in self.layers:
            last_layer_output = layer.forward(last_layer_output)
            # Do not forget to add bias for every layer
            if layer != self.layers[-1]:
                last_layer_output = np.insert(last_layer_output, 0, 1, axis=0)

        return last_layer_output

    def _compute_error(self, output, target):
        """
        Compute the total error of the network

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        self._err += self.error.calculate_error(target, output)

    def _update_weights(self, output, target):
        """
        Update the weights of the layers by propagating back the error
        """
        # MSE: Compute error wrt to the last layers output
        deriv = output - target

        for layer in reversed(self.layers):
            deriv = layer.computeDerivative(deriv)
            layer.updateWeights(self.learning_rate)
            deriv = np.delete(deriv, 0, axis=0)

    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """
        self._err = 0.0
        for img, label in zip(self.training_set.input, self.training_set.labels):
            output = self._feed_forward(img)
            self._compute_error(output, label)
            self._update_weights(output, label)
        self._err /= len(self.training_set.input)

    def _get_weights(self):
        """
        Get the encoder weights (after training)
        """
        return self.layers[:-1]
