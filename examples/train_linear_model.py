# coding : utf-8

from __future__ import print_function, absolute_import, division, unicode_literals
import sys
import logging
import numpy as np
from models import LogReg, SVM
import data_loader 

logging.basicConfig( format='%(message)s', level=logging.INFO )

# Set seed
seed = 123
np.random.seed( seed )

# Data load
split = True
standardize = True
normalize   = False
bias        = True

datafile_name = sys.argv[1]
if len( sys.argv ) >= 3:
    testfile_name = sys.argv[2]
else:
    testfile_name = None

X, Y, Xt, Yt = data_loader.load( datafile_name, testfile_name, split, 
                                 standardize, normalize, bias )

if Xt is not None:
    logging.info( 'train size: {0}, test size: {1}'.format( X.shape[0],Xt.shape[0] ) )
else:
    logging.info( 'train size: {0}'.format( X.shape[0] ) )
sys.stdout.flush()

# Build model and train.
model = u'svm'
input_dim = X.shape[1]
n_class = len(set(Y)) 

hparams = { 'shape'          : (input_dim,n_class),
            'wr'             : 1e-5,
            'eta'            : 1e-2,
            'momentum'       : 0.9, 
            'scale'          : 1.0, 
            'minibatch_size' : 100 }

if model == 'logreg':
    model = LogReg( **hparams )
elif model == 'svm':
    model = SVM( **hparams )
else:
    logging.log( logging.ERROR, 'invalid model: {0}'.format( model ) )
    sys.exit(-1)

model.fit( X, Y, 10, Xt, Yt, level=logging.INFO )

train_loss, train_acc = model.evaluate( X, Y )

logging.info( '- Result -' )
logging.info( 'train_loss: {0:5.4f}, train_acc: {1:4.3f}'\
              .format( train_loss, train_acc ) )

if Xt is not None:
    test_loss, test_acc  = model.evaluate( Xt,  Yt )
    logging.info( 'test_loss : {0:5.4f}, test_acc : {1:4.3f}'\
                  .format( test_loss, test_acc ) )

