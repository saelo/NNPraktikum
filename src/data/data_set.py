# -*- coding: utf-8 -*-

import numpy as np


class DataSet(object):
    """
    Representing train, valid or test sets

    Parameters
    ----------
    data : list
    oneHot : bool

    Attributes
    ----------
    input : list
    label : list
        A labels for the data given in `input`.
    one_hot : bool
    """

    def mk_one_hot(self, v, n):
        vec = np.zeros(n)
        vec[v] = 1
        return vec

    def __init__(self, data, one_hot=True):

        # The label of the digits is always the first fields
        self.input = 1.0 * data[:, 1:]/255
        self.label = data[:, 0]
        self.one_hot = one_hot

        if one_hot:
            self.label = np.array(list(map(lambda a: self.mk_one_hot(a, 10), self.label)))

    def __iter__(self):
        return self.input.__iter__()
