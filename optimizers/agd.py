# coding : utf-8

"""
Nesterov's accelerated gradient descent
"""

from __future__ import print_function, absolute_import, division, unicode_literals
from collections import OrderedDict
import numpy as np
import logging
import theano
import theano.tensor as T
import utils

class AGD( object ):
    """
    Nesterov's accelerated gradient descent with constant momentum and learning_rate 
    under minibath setting.
    Update rule: by
        Y. Bengio et al., Advances in Optimization Recurrent Networks. ICASSP, 2013.

        v_{t+1} = mu*v_{t} - eta*g(x_{t} + mu*v_{t}),
        x_{t+1} = x_{t} + v_{t+1},
    
    where g is the stochastic gradient. 
    Let p_{t} = x_{t} + mu*v_{t}, then we have see that the above update 
    is equivalent to

        v_{t+1} = mu*v_{t} - eta*g(p_{t}),
        p_{t+1} = p_{t} - eta*g(p_{t}) + mu*v_{t+1},

    because p_{t} - eta*g(p_{t}) = x_{t} + v_{t+1} = p_{t} - mu*v_{t} + v_{t+1} 
          = x_{t+1}.

    Though, x_T = p_T - mu*v_T = p_{T-1} - eta*g(p_{T-1}), we use p_T as the final point 
    because it is expected that v_T ~ 0 as optimization proceeds (T->\infty).
    """

    def __init__( self, model, eta=1e-2, momentum=0.9 ):
        """
        Initialize Nesterov's accelerated gradient descent.

        Arguments
        ---------
        model          : model object should be equiped with 
                         get_params, gradients, and get_symbols.
        eta            : Numerator of learning rate.
        momentum       : Nesterov's momentum.
        """

        self.model            = model
        self.__eta            = theano.shared( utils.numpy_floatX(eta) )
        self.__mu             = momentum
        self.__compile()

    def get_eta( self ):
        return self.__eta.get_value()

    def set_eta( self, eta ):
        self.__eta.set_value( utils.numpy_floatX(eta) )

    def show_eta( self ):
        logging.info( 'eta: 10.5'.format( self.__eta.get_value() ) )

    def __compile( self ):
        params = self.model.get_params()
        sgrads = self.model.gradients()

        # Shared variables for velocities.
        ves = [ theano.shared( 
            np.zeros( p.get_value().shape, dtype=theano.config.floatX ) ) 
                for p in params ]

        # Stochastic gradient.
        updates = OrderedDict()

        # Update velocities.
        new_ves = map( lambda (ve, sg) : self.__mu * ve - self.__eta * sg,
                       zip( ves, sgrads ) )
        updates.update( zip( ves, new_ves ) )       

        # AGD update.
        new_params = [ p - self.__eta * sg + self.__mu * new_ve 
                       for (p, sg, new_ve) 
                       in zip( params, sgrads, new_ves ) ]

        updates.update( zip( params, new_params ) )
      
        self.update_func = theano.function( inputs  = self.model.get_symbols(), 
                                            updates = updates )
                                                  
        # Reset function.
        updates = OrderedDict( [ ( ve, 0.*ve ) for ve in ves ] )
        self.reset_func = theano.function( inputs=[], updates=updates ) 
