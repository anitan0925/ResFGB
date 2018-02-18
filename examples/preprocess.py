# coding : utf-8

from __future__ import print_function, absolute_import, division, unicode_literals
import numpy as np
import theano

def add_bias( X, bias=1. ):
    n = X.shape[0]
    ones = bias * np.ones((n,1), dtype=theano.config.floatX )
    return np.concatenate( (X,ones), axis=1 )

def max_scale( X ):
    return np.max( np.sqrt( np.sum( X ** 2, axis=1 ) ) )    
    
def normalize_l2( X ):    
    n = X.shape[0]
    norms = np.sqrt( np.sum( X ** 2, axis=1 ) ).reshape(n,1)
    return X / norms
