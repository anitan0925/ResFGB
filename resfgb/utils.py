# coding : utf-8

from __future__ import print_function, absolute_import, division, unicode_literals
import numpy as np
import theano

def numpy_floatX( data ):
    return np.asarray( data, dtype=theano.config.floatX )

def minibatches( minibatch_size, X, Y=None, shuffle=False ):
    indices = np.arange( X.shape[0] )
    if shuffle:
        np.random.shuffle( indices )

    for start in range( 0, X.shape[0], minibatch_size ):
        minibatch_indices = indices[ start : start+minibatch_size ]
        if Y is None:
            yield X[minibatch_indices]
        else:
            yield X[minibatch_indices], Y[minibatch_indices]

def minibatches2( minibatch_size, X, X2, Y, shuffle=False ):
    indices = np.arange( X.shape[0] )
    if shuffle:
        np.random.shuffle( indices )

    for start in range( 0, X.shape[0], minibatch_size ):
        minibatch_indices = indices[ start : start+minibatch_size ]
        yield X[minibatch_indices], X2[minibatch_indices], Y[minibatch_indices]

def minibatch_indices( minibatch_size, n, shuffle=False ):
    indices = np.arange( n )
    if shuffle:
        np.random.shuffle( indices )

    for start in range( 0, n, minibatch_size ):
        minibatch_indices = indices[ start : start+minibatch_size ]
        yield minibatch_indices

