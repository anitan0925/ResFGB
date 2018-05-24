# coding : utf-8

"""
Support vector machine for multiclass classificcation problems.
"""

from __future__ import print_function, absolute_import, division, unicode_literals
import sys
import time
from logging import getLogger
import numpy as np
import theano
import theano.tensor as T
from resfgb.utils import minibatches
from resfgb.models import layers as L
from resfgb.models.classifier import Classifier
from resfgb.optimizers import AGD

logger = getLogger(__name__)


class SVM(Classifier):
    def __init__(self, shape, bias=True, wr=0, eta=1e-2, momentum=0.9, gamma=1e+0,
                 scale=1., minibatch_size=10, seed=99):
        """
        shape          : tuple of integers.
                         Dimension and the number of classes
        bias           : flag for whether to use bias or not.
        wr             : float.
                         The L2-regularization paremter.
        opt_params     : dictionary.
        minibatch_size : integer.
                         Minibatch size to calcurate stochastic gradient.
        seed           : integer.
                         Seed for random module.
        """
        super(SVM, self).__init__(eta, scale, minibatch_size, seed)

        self.show_param(shape, wr, eta, momentum, scale, minibatch_size, seed)

        # input symbols.
        self.Z = T.matrix(dtype=theano.config.floatX)
        self.Y = T.ivector()
        self.symbols = [self.Z, self.Y]

        # parameters.
        W = L.linear_param(shape[0], shape[1], scale=5e-2)
        b = L.zeros_param(shape[1])
        if bias:
            self.params = [b, W]
        else:
            self.params = [W]

        # functions.
        A = L.FullConnect(self.Z, self.params)  # (n,K), K is the number of classes.
        margin = A[T.arange(self.Y.shape[0]), self.Y][:, None] - A  # (n,K)
        self.loss = T.mean(T.sum(T.nnet.softplus(gamma - margin), axis=1))
        self.pred = T.argmax(A, axis=1)

        if wr > 0:
            self.wr = wr
            if bias:
                self.reg = 0.5 * wr * T.sum(self.params[1]**2)
            else:
                self.reg = 0.5 * wr * T.sum(self.params[0]**2)
        else:
            self.wr = 0
            self.reg = 0

        self.sgrad = T.grad(cost=self.loss + self.reg, wrt=self.params)

        # compile.
        self.compile()

        # optimizer.
        self.optimizer = AGD(self, eta=eta, momentum=momentum)

    def show_param(self, shape, wr, eta, momentum, scale,
                   minibatch_size, seed):
        logger.info('{0:<5}{1:^26}{2:>5}'.format('-' * 5, 'SVM setting', '-' * 5))
        logger.info('{0:<15}{1:>21}'.format('dim', shape[0]))
        logger.info('{0:<15}{1:>21}'.format('n_class', shape[1]))
        logger.info('{0:<15}{1:>21.7}'.format('wr', wr))
        logger.info('{0:<15}{1:>21.7f}'.format('eta', eta))
        logger.info('{0:<15}{1:>21.7f}'.format('momentum', momentum))
        logger.info('{0:<15}{1:>21.7f}'.format('scale', scale))
        logger.info('{0:<15}{1:>21}'.format('minibatch_size', minibatch_size))
        logger.info('{0:<15}{1:>21}'.format('seed', seed))

    def compile(self):
        self.predict = theano.function([self.Z], self.pred)
        self.loss_func = theano.function([self.Z, self.Y], self.loss)
        self.reg_func = theano.function([], self.reg)
