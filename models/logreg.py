# coding : utf-8

from __future__ import print_function, absolute_import, division, unicode_literals
import sys
import time
import logging
import numpy as np
import theano
import theano.tensor as T
from utils import minibatches
import models.layers as L
from models.classifier import Classifier
from optimizers import AGD

class LogReg( Classifier ):
    def __init__( self, shape, wr=None, eta=1e-2, momentum=0.9, scale=1., 
                  minibatch_size=10, seed=99 ):
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
        super( LogReg, self ).__init__( eta, scale, minibatch_size, seed )

        self.show_param( shape, wr, eta, momentum, scale, minibatch_size, seed )

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
        self.optimizer  = AGD( self, eta=eta, momentum=momentum )

    def show_param( self, shape, wr, eta, momentum, scale,
                    minibatch_size, seed ):
        logging.info('- LogReg hyperparameters -')
        logging.info( '{0:<15}{1:>11}'.format( 'dim', shape[0] ) )
        logging.info( '{0:<15}{1:>11}'.format( 'n_class', shape[1] ) )
        logging.info( '{0:<15}{1:>11.7}'.format( 'wr', wr ) )
        logging.info( '{0:<15}{1:>11.7f}'.format( 'eta', eta ) )
        logging.info( '{0:<15}{1:>11.7f}'.format( 'momentum', momentum ) )
        logging.info( '{0:<15}{1:>11.7f}'.format( 'scale', scale ) )
        logging.info( '{0:<15}{1:>11}'.format( 'minibatch_size', minibatch_size ) )
        logging.info( '{0:<15}{1:>11}'.format( 'seed', seed ) )

    def compile( self ):
        self.predict   = theano.function( [self.Z], self.pred )
        self.loss_func = theano.function( [self.Z,self.Y], self.loss )
        self.reg_func  = theano.function( [], self.reg )        
