#!/usr/bin/env python
# -*- coding: utf-8 -*-

from data.mnist_seven import MNISTSeven
from model.stupid_recognizer import StupidRecognizer
from model.perceptron import Perceptron
from model.logistic_regression import LogisticRegression
from report.evaluator import Evaluator
from report.performance_plot import PerformancePlot


def main():
    data = MNISTSeven("../data/mnist_seven.csv", 3000, 1000, 1000)

    # Train the classifiers
    print("=========================")
    print("Training..")

    myStupidClassifier = StupidRecognizer(data.training_set,
                                          data.validation_set,
                                          data.test_set)

    print("\nStupid Classifier has been training..")
    myStupidClassifier.train()
    print("Done..")
    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    stupidPred = myStupidClassifier.evaluate()

    myPerceptronClassifier = Perceptron(data.training_set,
                                        data.validation_set,
                                        data.test_set,
                                        learning_rate=0.005,
                                        epochs=10)

    print("\nPerceptron has been training..")
    myPerceptronClassifier.train()
    print("Done..")
    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    perceptronPred = myPerceptronClassifier.evaluate()

    myLRClassifier = LogisticRegression(data.training_set,
                                        data.validation_set,
                                        data.test_set,
                                        learning_rate=0.005,
                                        epochs=30)

    print("\nLogistic Regression has been training..")
    myLRClassifier.train()
    print("Done..")
    # Do the recognizer
    # Explicitly specify the test set to be evaluated
    lrPred = myLRClassifier.evaluate()

    # Report the result
    print("=========================")
    evaluator = Evaluator()

    print("Result of the stupid recognizer:")
    # evaluator.printComparison(data.testSet, stupidPred)
    evaluator.printAccuracy(data.test_set, stupidPred)

    print("\nResult of the Perceptron recognizer (on test set):")
    # evaluator.printComparison(data.testSet, perceptronPred)
    evaluator.printAccuracy(data.test_set, perceptronPred)

    print("\nResult of the Logistic Regression recognizer (on test set):")
    # evaluator.printComparison(data.testSet, perceptronPred)
    evaluator.printAccuracy(data.test_set, lrPred)

    # Draw
    plot = PerformancePlot("Logistic Regression")
    plot.draw_performance_epoch(myLRClassifier.performances,
                                myLRClassifier.epochs)

if __name__ == '__main__':
    main()
