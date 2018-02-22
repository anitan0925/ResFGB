# config : utf-8

from __future__ import print_function, absolute_import, division, unicode_literals
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from .preprocess import max_scale, add_bias, normalize_l2
if sys.version_info[0] == 2:
    import cPickle
else:
    import _pickle as cPickle

def load( trainfile_name, testfile_name, 
          split, standardize, normalize, bias, 
          train_ratio=0.8, shuffle=True ):
    
    fin = open( trainfile_name, u'rb' )
    [X,Y] = cPickle.load(fin)
    fin.close()

    indices = np.arange( X.shape[0] )
    if shuffle:
        np.random.shuffle( indices )
    X = X[indices]
    Y = Y[indices]

    if testfile_name is not None:
        fin = open( testfile_name, u'rb' )
        [Xt,Yt] = cPickle.load(fin)
        fin.close()
    elif split:
        train_size = int( train_ratio * X.shape[0] )
        indices = range( X.shape[0] )
        if shuffle:
            np.random.shuffle( indices )
        train_mask = indices[:train_size]
        test_mask  = indices[train_size:]
        Xt = X[test_mask]
        Yt = Y[test_mask]
        X  = X[train_mask]
        Y  = Y[train_mask]
    else:
        Xt = None
        Yt = None
        
    if standardize:
        scaler = StandardScaler()
        scaler.fit( X )
        X  = scaler.transform( X )
        if Xt is not None:
            Xt = scaler.transform( Xt )

    if bias:
        X  = add_bias(X)        
        if Xt is not None:
            Xt = add_bias(Xt)

    if normalize:
        X  = normalize_l2( X )
        if Xt is not None:
            Xt = normalize_l2( Xt )

    if Xt is not None:
        return X.astype(np.float32), Y.astype(np.int32),\
            Xt.astype(np.float32), Yt.astype(np.int32)
    else:
        return X.astype(np.float32), Y.astype(np.int32), None, None
