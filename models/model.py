# coding : utf-8

from __future__ import print_function, absolute_import, division, unicode_literals
import numpy as np

class Model( object ):
    def __init__( self, seed ):
        np.random.seed( seed )

    def gradients( self ):
        return self.sgrad

    def get_params( self, real_f=False ):
        if real_f:
            return [ p.get_value() for p in self.params ]
        else:
            return self.params

    def set_params( self, param, real_f=True ):
        if real_f:
            map( lambda (p_,p) : p.set_value(p_), zip(param,self.params) )
        else:
            map( lambda (p_,p) : p.set_value(p_.get_value()), zip(param,self.params) )
  
    def get_symbols( self ):
        return self.symbols

    def save_params( self ):
        self.__saved_params__ = self.get_params( real_f=True )

    def load_params( self ):
        self.set_params( self.__saved_params__, real_f=True )
