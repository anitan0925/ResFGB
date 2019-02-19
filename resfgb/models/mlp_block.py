# coding : utf-8

"""
Multilayer perceptron for multidimensional regression.
"""

from __future__ import print_function, absolute_import, division, unicode_literals
import sys
import time
from logging import getLogger, DEBUG, ERROR
import numpy as np
import theano
import theano.tensor as T
from resfgb.utils import minibatches
from resfgb.models import layers as L
from resfgb.models.regressor import Regressor, Regressor2
from resfgb.optimizers import AGD

logger = getLogger(__name__)

class MLPBlock(Regressor):
    def __init__(self, shape, wr=0, eta=1e-2, momentum=0.9, scale=1.,
                 minibatch_size=10, eval_iters=1000, seed=99,
                 log_level=DEBUG):
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
        self.symbols = [self.X, self.Z, self.Y]

        # parameters.
        self.params = []
        for l in range(len(shape) - 1):
            b = L.zeros_param(shape[l + 1])
            W = L.linear_param(shape[l], shape[l + 1], scale=5e-2)
            self.params.extend([b, W])

        # functions.
        normalize = True # test
        if normalize:
            Z = L.normalize( self.Z, 1e-4 ) * float( shape[0] )
        else:
            Z = self.Z
            
        for l in range(0, len(shape) - 1):
            b = self.params[2 * l]
            W = self.params[2 * l + 1]
            if l == len(shape) - 2:
                Z = L.Act(L.FullConnect(Z, [b, W]), 'tanh') 
            else:
                Z = L.Act(L.FullConnect(Z, [b, W]), 'relu') 
        self.output = Z
        self.loss = L.Loss(self.output, self.Y, 'huber')
        # self.loss = L.Loss(self.output, self.Y, 'abs')
        # self.loss = L.Loss(self.output, self.Y, 'squared_error')

        if wr > 0:
            self.wr = wr
            val = 0
            for l in range(1, len(self.params), 2):
                val += T.sum(self.params[l]**2)
            self.reg = 0.5 * wr * val
        else:
            logger.log(ERROR,
                       'negative regularization parameter is given: {0}'.format(wr))
            sys.exit(-1)

        self.sgrad = T.grad(cost=self.loss + self.reg, wrt=self.params)

        # compile.
        self.compile()

        # optimizer.
        self.optimizer = AGD(self, eta=eta, momentum=momentum)

    def show_param(self, shape, wr, eta, momentum, scale,
                   minibatch_size, eval_iters, seed):
        logger.info('{0:<5}{1:^26}{2:>5}'.format('-' * 5, 'MLPBlock setting', '-' * 5))
        logger.info('{0:<5}{1:>31}'.format('shape', ' '.join(map(str, shape))))
        logger.info('{0:<15}{1:>21.7}'.format('wr', wr))
        logger.info('{0:<15}{1:>21.7f}'.format('eta', eta))
        logger.info('{0:<15}{1:>21.7f}'.format('momentum', momentum))
        logger.info('{0:<15}{1:>21.7f}'.format('scale', scale))
        logger.info('{0:<15}{1:>21}'.format('minibatch_size', minibatch_size))
        logger.info('{0:<15}{1:>21}'.format('eval_iters', eval_iters))
        logger.info('{0:<15}{1:>21}'.format('seed', seed))

    def compile(self):
        self.predict = theano.function([self.Z], self.output)
        self.loss_func = theano.function([self.Z, self.Y], self.loss)
        self.reg_func = theano.function([], self.reg)

class MLPBlock2(Regressor2):
    def __init__(self, shape, wr=0, eta=1e-2, momentum=0.9, scale=1.,
                 minibatch_size=10, eval_iters=1000, seed=99,
                 log_level=DEBUG):
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
        super(MLPBlock2, self).__init__(eta, scale, minibatch_size, eval_iters, seed,
                                        log_level)

        self.show_param(shape, wr, eta, momentum, scale, minibatch_size,
                        eval_iters, seed)

        # input symbols.
        self.X = T.matrix(dtype=theano.config.floatX)
        self.Z = T.matrix(dtype=theano.config.floatX)
        self.Y = T.matrix(dtype=theano.config.floatX)
        self.symbols = [self.X, self.Z, self.Y]

        # parameters.
        self.params = []
        for l in range(len(shape) - 1):
            b = L.zeros_param(shape[l + 1])
            W = L.linear_param(shape[l], shape[l + 1], scale=5e-2)
            b2 = L.zeros_param(shape[l + 1])
            W2 = L.linear_param(shape[l], shape[l + 1], scale=5e-2)
            self.params.extend([b, W, b2, W2])

        # functions.
        normalize = False # test
        X = self.X
        if normalize:
            Z = L.normalize( self.Z, 1e-4 ) * float( shape[0] )
            Z2 = L.normalize( self.X, 1e-4 ) * float( shape[0] )
        else:
            Z = self.Z
            Z2 = self.X

        for l in range(0, len(shape) - 1):
            b = self.params[4 * l]
            W = self.params[4 * l + 1]
            b2 = self.params[4 * l + 2]
            W2 = self.params[4 * l + 3]
            if l == len(shape) - 2:
                Z = L.Act(L.FullConnect(Z, [b, W]), 'tanh')
                Z2 = L.Act(L.FullConnect(Z2, [b2, W2]), 'tanh' )
            else:
                Z = L.Act(L.FullConnect(Z, [b, W]), 'relu') 
                Z2 = L.Act(L.FullConnect(Z2, [b2, W2]), 'relu' )

        self.output_1 = Z  # receive the output of previous layer.
        self.output_2 = Z2 # receive input data directly.
        self.output = Z + Z2
        self.loss = L.Loss(self.output, self.Y, 'inner_prod')
        # self.loss = L.Loss(self.output, self.Y, 'huber')
        # self.loss = L.Loss(self.output, self.Y, 'abs')
        # self.loss = L.Loss(self.output, self.Y, 'squared_error')

        if wr > 0:
            self.wr = wr
            val = 0
            for l in range(1, len(self.params), 2):
                val += T.sum(self.params[l]**2)
            self.reg = 0.5 * wr * val
        else:
            logger.log(ERROR,
                       'negative regularization parameter is given: {0}'.format(wr))
            sys.exit(-1)

        self.sgrad = T.grad(cost=self.loss + self.reg, wrt=self.params)

        # compile.
        self.compile()

        # optimizer.
        self.optimizer = AGD(self, eta=eta, momentum=momentum)

    def show_param(self, shape, wr, eta, momentum, scale,
                   minibatch_size, eval_iters, seed):
        logger.info('{0:<5}{1:^26}{2:>5}'.format('-' * 5, 'MLPBlock setting', '-' * 5))
        logger.info('{0:<5}{1:>31}'.format('shape', ' '.join(map(str, shape))))
        logger.info('{0:<15}{1:>21.7}'.format('wr', wr))
        logger.info('{0:<15}{1:>21.7f}'.format('eta', eta))
        logger.info('{0:<15}{1:>21.7f}'.format('momentum', momentum))
        logger.info('{0:<15}{1:>21.7f}'.format('scale', scale))
        logger.info('{0:<15}{1:>21}'.format('minibatch_size', minibatch_size))
        logger.info('{0:<15}{1:>21}'.format('eval_iters', eval_iters))
        logger.info('{0:<15}{1:>21}'.format('seed', seed))

    def compile(self):
        self.predict = theano.function([self.X, self.Z], self.output)
        self.predict_1 = theano.function([self.Z], self.output_1)
        self.predict_2 = theano.function([self.X], self.output_2)        
        self.loss_func = theano.function([self.X, self.Z, self.Y], self.loss)
        self.reg_func = theano.function([], self.reg)
