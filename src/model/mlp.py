
import numpy as np

# from util.activation_functions import Activation
from model.logistic_layer import LogisticLayer
from model.classifier import Classifier
from util.loss_functions import CrossEntropyError
from sklearn.metrics import accuracy_score


class MultilayerPerceptron(Classifier):
    """
    A multilayer perceptron used for classification
    """

    def __init__(self, train, valid, test, layers=None, input_weights=None,
                 output_task='classification', output_activation='softmax',
                 cost='crossentropy', learning_rate=0.01, epochs=50):

        """
        A digit-7 recognizer based on logistic regression algorithm

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
        self.output_task = output_task  # Either classification or regression
        self.output_activation = output_activation
        self.cost = cost

        self.training_set = train
        self.validation_set = valid
        self.test_set = test

        # Record the performance of each epoch for later usages
        # e.g. plotting, reporting..
        self.performances = []

        self.layers = layers
        self.input_weights = input_weights

        # add bias values ("1"s) at the beginning of all data sets
        self.training_set.input = np.insert(self.training_set.input, 0, 1,
                                            axis=1)
        self.validation_set.input = np.insert(self.validation_set.input, 0, 1,
                                              axis=1)
        self.test_set.input = np.insert(self.test_set.input, 0, 1, axis=1)

        # Build up the network from specific layers
        # Here is an example of a MLP acting like the Logistic Regression
        if not layers:
            hidden_layer_size = 128
            self.layers = []
            self.layers.append(LogisticLayer(self.training_set.input.shape[1], hidden_layer_size, None, "sigmoid", False))
            self.layers.append(LogisticLayer(hidden_layer_size, 10, None, self.output_activation, True))

    def _get_layer(self, layer_index):
        return self.layers[layer_index]

    def _get_input_layer(self):
        return self.get_layer(0)

    def _get_output_layer(self):
        return self.get_layer(-1)

    def _feed_forward(self, inp):
        """
        Do feed forward through the layers of the network

        Parameters
        ----------
        inp : ndarray
            a numpy array containing the input of the layer

        # Here you have to propagate forward through the layers
        # And remember the activation values of each layer
        """
        cur = inp

        for layer in self.layers:
            cur = layer.forward(cur)

        self._output = cur
        return self._output

    def _compute_error(self, target):
        """
        Compute the total error of the network

        Returns
        -------
        ndarray :
            a numpy array (1,nOut) containing the output of the layer
        """
        if self.cost == 'crossentropy':
            loss = CrossEntropyError()
        else:
            raise "Unsupported cost function"

        return loss.calculate_error(target, self._output)

    def _update_weights(self, label):
        """
        Update the weights of the layers by propagating back the error
        """
        deriv = self.layers[-1].computeDerivative(None, label)
        self.layers[-1].updateWeights(self.learning_rate)

        for layer in self.layers[-2::-1]:
            deriv = layer.computeDerivative(deriv)
            layer.updateWeights(self.learning_rate)

    def train(self, verbose=True):
        """Train the Multi-layer Perceptrons

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """

        # Run the training "epochs" times, print out the logs
        for epoch in range(self.epochs):
            if verbose:
                print("Training epoch {0}/{1}.."
                      .format(epoch + 1, self.epochs))

            self._train_one_epoch()

            if verbose:
                accuracy = accuracy_score(self.validation_set.label,
                                          self.evaluate(self.validation_set))
                # Record the performance of each epoch for later usages
                # e.g. plotting, reporting..
                self.performances.append(accuracy)
                print("Accuracy on validation: {0:.2f}%"
                      .format(accuracy * 100))
                print("Current training error: {0:.2f}".format(self._error))
                print("-----------------------------")


    def _train_one_epoch(self):
        """
        Train one epoch, seeing all input instances
        """

        self._error = 0.0
        for img, label in zip(self.training_set.input,
                              self.training_set.label):

            # Convert the label to a one hot vector
            label_one_hot = np.zeros(10)
            label_one_hot[label] = 1

            # Do a forward pass to calculate the output and the error
            self._feed_forward(img)
            error = self._compute_error(label_one_hot)
            self._error += error
            self._update_weights(label_one_hot)

        self._error /= len(self.training_set.input)

    def classify(self, test_instance):
        # Classify an instance given the model of the classifier
        # You need to implement something here
        output = self._feed_forward(test_instance)
        return np.argmax(output)

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.test_set.input
        # Once you can classify an instance, just use map for all of the test set.
        return list(map(self.classify, test))

    def __del__(self):
        # Remove the bias from input data
        self.training_set.input = np.delete(self.training_set.input, 0, axis=1)
        self.validation_set.input = np.delete(self.validation_set.input, 0,
                                              axis=1)
        self.test_set.input = np.delete(self.test_set.input, 0, axis=1)
