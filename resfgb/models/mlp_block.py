# coding : utf-8

"""
Multilayer perceptron for multidimensional regression.
"""

from __future__ import print_function, absolute_import, division, unicode_literals
import sys
import time
import logging
import numpy as np
import theano
import theano.tensor as T
from resfgb.utils import minibatches
from resfgb.models import layers as L
from resfgb.models.regressor import Regressor
from resfgb.optimizers import AGD


class MLPBlock(Regressor):
    def __init__(self, shape, wr=0, eta=1e-2, momentum=0.9, scale=1.,
                 minibatch_size=10, eval_iters=1000, seed=99,
                 log_level=logging.DEBUG):
        """
        shape          : tuple of integers.
                         Dimension and the number of classes
        wr             : float.
                         The L2-regularization paremter.
        opt_params     : dictionary.
        minibatch_size : integer.
                         Minibatch size to calcurate stochastic gradient.
        seed           : integer.
                         Seed for random module.
        """
        super(MLPBlock, self).__init__(eta, scale, minibatch_size, eval_iters, seed,
                                       log_level)

        self.show_param(shape, wr, eta, momentum, scale, minibatch_size,
                        eval_iters, seed)

        # input symbols.
        self.Z = T.matrix(dtype=theano.config.floatX)
        self.Y = T.matrix(dtype=theano.config.floatX)
        self.symbols = [self.Z, self.Y]

        # parameters.
        self.params = []
        for l in range(len(shape) - 1):
            b = L.zeros_param(shape[l + 1])
            W = L.linear_param(shape[l], shape[l + 1], scale=5e-2)
            self.params.extend([b, W])

        # functions.
        Z = self.Z
        for l in range(0, len(shape) - 1):
            b = self.params[2 * l]
            W = self.params[2 * l + 1]
            Z = L.Act(L.FullConnect(Z, [b, W]), 'relu')
        self.output = Z
        self.loss = L.Loss(self.output, self.Y, 'squared_error')

        if wr > 0:
            self.wr = wr
            val = 0
            for l in range(1, len(self.params), 2):
                val += T.sum(self.params[l]**2)
            self.reg = 0.5 * wr * val
        else:
            logging.log(logging.ERROR,
                        'negative regularization parameter is given: {0}'.format(wr))
            sys.exit(-1)

        self.sgrad = T.grad(cost=self.loss + self.reg, wrt=self.params)

        # compile.
        self.compile()

        # optimizer.
        self.optimizer = AGD(self, eta=eta, momentum=momentum)

    def show_param(self, shape, wr, eta, momentum, scale,
                   minibatch_size, eval_iters, seed):
        logging.info('{0:<5}{1:^26}{2:>5}'.format('-' * 5, 'MLPBlock setting', '-' * 5))
        logging.info('{0:<5}{1:>31}'.format('shape', ' '.join(map(str, shape))))
        logging.info('{0:<15}{1:>21.7}'.format('wr', wr))
        logging.info('{0:<15}{1:>21.7f}'.format('eta', eta))
        logging.info('{0:<15}{1:>21.7f}'.format('momentum', momentum))
        logging.info('{0:<15}{1:>21.7f}'.format('scale', scale))
        logging.info('{0:<15}{1:>21}'.format('minibatch_size', minibatch_size))
        logging.info('{0:<15}{1:>21}'.format('eval_iters', eval_iters))
        logging.info('{0:<15}{1:>21}'.format('seed', seed))

    def compile(self):
        self.predict = theano.function([self.Z], self.output)
        self.loss_func = theano.function([self.Z, self.Y], self.loss)
        self.reg_func = theano.function([], self.reg)
