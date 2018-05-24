# coding : utf-8

"""
Logistic regression for multiclass classificcation problems.
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
from resfgb.models.classifier import Classifier
from resfgb.optimizers import AGD

logger = getLogger(__name__)


class LogReg(Classifier):
    def __init__(self, shape, bias=True, wr=0, eta=1e-2, momentum=0.9, scale=1.,
                 minibatch_size=10, eval_iters=1000, seed=99,
                 log_level=DEBUG):
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
        super(LogReg, self).__init__(eta, scale, minibatch_size, eval_iters, seed,
                                     log_level)

        self.show_param(shape, wr, eta, momentum, scale, minibatch_size,
                        eval_iters, seed)

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
        output = L.Act(L.FullConnect(self.Z, self.params), u'softmax')
        self.pred = T.argmax(output, axis=1)
        self.loss = L.Loss(output, self.Y)
        if wr > 0:
            self.wr = wr
            if bias:
                self.reg = 0.5 * wr * T.sum(self.params[1]**2)
            else:
                self.reg = 0.5 * wr * T.sum(self.params[0]**2)
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
        logger.info('{0:<5}{1:^26}{2:>5}'.format('-' * 5, 'LogReg setting', '-' * 5))
        logger.info('{0:<15}{1:>21}'.format('dim', shape[0]))
        logger.info('{0:<15}{1:>21}'.format('n_class', shape[1]))
        logger.info('{0:<15}{1:>21.7}'.format('wr', wr))
        logger.info('{0:<15}{1:>21.7f}'.format('eta', eta))
        logger.info('{0:<15}{1:>21.7f}'.format('momentum', momentum))
        logger.info('{0:<15}{1:>21.7f}'.format('scale', scale))
        logger.info('{0:<15}{1:>21}'.format('minibatch_size', minibatch_size))
        logger.info('{0:<15}{1:>21}'.format('eval_iters', eval_iters))
        logger.info('{0:<15}{1:>21}'.format('seed', seed))

    def compile(self):
        self.predict = theano.function([self.Z], self.pred)
        self.loss_func = theano.function([self.Z, self.Y], self.loss)
        self.reg_func = theano.function([], self.reg)
