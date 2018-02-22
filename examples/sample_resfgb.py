# coding : utf-8

from __future__ import print_function, absolute_import, division, unicode_literals
import sys
import logging
import numpy as np
from resfgb.models import ResFGB, LogReg, SVM, get_hyperparams
from scripts import sample_data

logging.basicConfig( format='%(message)s', level=logging.INFO )

# Set seed
seed = 123
np.random.seed( seed )

# Get data
X, Y, Xt, Yt = sample_data.get_ijcnn1()
n_train = int( 0.8 * X.shape[0] )
Xv, Yv = X[n_train:], Y[n_train:]
X, Y = X[:n_train], Y[:n_train]

logging.info( 'train size: {0}, validation size: {1}, test size: {2}'\
              .format( X.shape[0],Xv.shape[0],Xt.shape[0] ) )

# Build model and train.
(n_data, input_dim) = X.shape
n_class = len( set(Y) | set(Yv) | set(Yt) )

hparams = get_hyperparams( n_data, input_dim, n_class )
hparams['model_hparams']['wr']           = 1e-5
hparams['model_hparams']['max_epoch']    = 30
hparams['resblock_hparams']['wr']        = 1e-5
hparams['resblock_hparams']['max_epoch'] = 20
hparams['fg_eta']    = 1e-1
hparams['max_iters'] = 30
hparams['seed']      = seed

model = ResFGB( **hparams )
best_iters,_ ,_ = model.fit( X, Y, Xv, Yv, use_best_iter=True )

train_loss, train_acc = model.evaluate( X, Y )

logging.info( '- Result -' )
logging.info( 'train_loss: {0:5.4f}, train_acc: {1:4.3f}'\
              .format( train_loss, train_acc ) )

if Xt is not None:
    test_loss, test_acc  = model.evaluate( Xt,  Yt )
    logging.info( 'test_loss : {0:5.4f}, test_acc : {1:4.3f}'\
                  .format( test_loss, test_acc ) )

