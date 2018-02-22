# coding : utf-8

"""
Logistic regression for multiclass classificcation problems.
"""

from __future__ import print_function, absolute_import, division, unicode_literals
import sys, time, logging
import numpy as np
import theano
import theano.tensor as T
from resfgb.utils import minibatches
from resfgb.models import layers as L
from resfgb.models.classifier import Classifier
from resfgb.optimizers import AGD

class LogReg( Classifier ):
    def __init__( self, shape, wr=0, eta=1e-2, momentum=0.9, scale=1., 
                  minibatch_size=10, eval_iters=1000, seed=99,
                  log_level=logging.DEBUG ):
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
        super( LogReg, self ).__init__( eta, scale, minibatch_size, eval_iters, seed,
                                        log_level )

        self.show_param( shape, wr, eta, momentum, scale, minibatch_size, 
                         eval_iters, seed )

        # input symbols.
        self.Z  = T.matrix( dtype=theano.config.floatX )
        self.Y  = T.ivector()
        self.symbols = [ self.Z, self.Y ]

        # parameters.
        bias = False
        W = L.linear_param( shape[0], shape[1], scale=5e-2 )
        b = L.zeros_param( shape[1] )
        if bias:
            self.params = [b,W]
        else:
            self.params = [W]

        # functions.
        output    = L.Act( L.FullConnect( self.Z, self.params ), u'softmax' )
        self.pred = T.argmax( output, axis=1 )
        self.loss = L.Loss( output, self.Y )
        if wr > 0:
            self.wr = wr
            if bias:
                self.reg = 0.5 * wr * T.sum( self.params[1]**2 )
            else:
                self.reg = 0.5 * wr * T.sum( self.params[0]**2 )
        else:
            self.wr = 0
            self.reg = 0

        self.sgrad = T.grad( cost=self.loss + self.reg, wrt=self.params )

        # compile.
        self.compile()

        # optimizer.
        self.optimizer = AGD( self, eta=eta, momentum=momentum )

    def show_param( self, shape, wr, eta, momentum, scale,
                    minibatch_size, eval_iters, seed ):
        logging.info( '{0:<5}{1:^26}{2:>5}'.format( '-'*5, 'LogReg setting', '-'*5 ) )
        logging.info( '{0:<15}{1:>21}'.format( 'dim', shape[0] ) )
        logging.info( '{0:<15}{1:>21}'.format( 'n_class', shape[1] ) )
        logging.info( '{0:<15}{1:>21.7}'.format( 'wr', wr ) )
        logging.info( '{0:<15}{1:>21.7f}'.format( 'eta', eta ) )
        logging.info( '{0:<15}{1:>21.7f}'.format( 'momentum', momentum ) )
        logging.info( '{0:<15}{1:>21.7f}'.format( 'scale', scale ) )
        logging.info( '{0:<15}{1:>21}'.format( 'minibatch_size', minibatch_size ) )
        logging.info( '{0:<15}{1:>21}'.format( 'eval_iters', eval_iters ) )
        logging.info( '{0:<15}{1:>21}'.format( 'seed', seed ) )

    def compile( self ):
        self.predict   = theano.function( [self.Z], self.pred )
        self.loss_func = theano.function( [self.Z,self.Y], self.loss )
        self.reg_func  = theano.function( [], self.reg )        
