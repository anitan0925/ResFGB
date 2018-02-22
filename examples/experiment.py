# coding : utf-8

from __future__ import print_function, absolute_import, division, unicode_literals
import sys, logging
import numpy as np
from resfgb.models import ResFGB, LogReg, SVM, get_hyperparams
from scripts import sample_data

dataset = sys.argv[1]
logging.basicConfig( format='%(message)s', level=logging.INFO )

# Set seed
seed = 123
np.random.seed( seed )

# Get data
if dataset == 'letter':
    X, Y, Xt, Yt = sample_data.get_letter()
elif dataset == 'usps':
    X, Y, Xt, Yt = sample_data.get_usps()
elif dataset == 'ijcnn1':
    X, Y, Xt, Yt = sample_data.get_ijcnn1()
elif dataset == 'mnist':
    X, Y, Xt, Yt = sample_data.get_mnist()
elif dataset == 'covtype':
    X, Y, Xt, Yt = sample_data.get_covtype()

logging.info( 'train size: {0}, test size: {1}'\
              .format( X.shape[0],Xt.shape[0] ) )

# Build model and train.
(n_data, input_dim) = X.shape
n_class = len( set(Y) | set(Yt) )

hparams = get_hyperparams( n_data, input_dim, n_class )

if dataset == 'letter':
    hparams['model_hparams']['wr']            = 1e-6
    hparams['model_hparams']['max_epoch']     = 100
    hparams['model_hparams']['early_stop']    = -1
    hparams['resblock_hparams']['shape']      = (input_dim,1000,1000,1000,input_dim)
    hparams['resblock_hparams']['wr']         = 1e-6
    hparams['resblock_hparams']['max_epoch']  = 100
    hparams['resblock_hparams']['early_stop'] = 10
    hparams['fg_eta']    = 1e-2
    hparams['max_iters'] = 21

elif dataset == 'usps':
    hparams['model_hparams']['wr']            = 1e-4
    hparams['model_hparams']['max_epoch']     = 200
    hparams['model_hparams']['early_stop']    = -1
    hparams['resblock_hparams']['shape']      = (input_dim,1000,1000,1000,input_dim)
    hparams['resblock_hparams']['wr']         = 1e-4
    hparams['resblock_hparams']['max_epoch']  = 100
    hparams['resblock_hparams']['early_stop'] = 10
    hparams['fg_eta']    = 1e+0
    hparams['max_iters'] = 1

elif dataset == 'ijcnn1':
    hparams['model_hparams']['wr']            = 1e-5
    hparams['model_hparams']['max_epoch']     = 50
    hparams['model_hparams']['early_stop']    = -1
    hparams['resblock_hparams']['shape']      = (input_dim,100,100,100,100,input_dim)
    hparams['resblock_hparams']['wr']         = 1e-5
    hparams['resblock_hparams']['max_epoch']  = 20
    hparams['resblock_hparams']['early_stop'] = 10
    hparams['fg_eta']    = 1e-1
    hparams['max_iters'] = 28

elif dataset == 'mnist':
    hparams['model_hparams']['wr']            = 1e-6
    hparams['model_hparams']['max_epoch']     = 20
    hparams['model_hparams']['early_stop']    = -1
    hparams['resblock_hparams']['shape']      = (input_dim,1000,1000,1000,input_dim)
    hparams['resblock_hparams']['wr']         = 1e-6
    hparams['resblock_hparams']['max_epoch']  = 20
    hparams['resblock_hparams']['early_stop'] = 10
    hparams['fg_eta']    = 1e-1
    hparams['max_iters'] = 8

elif dataset == 'covtype':
    hparams['model_hparams']['wr']            = 1e-6
    hparams['model_hparams']['max_epoch']     = 10
    hparams['model_hparams']['early_stop']    = -1
    hparams['resblock_hparams']['shape']      = (input_dim,1000,1000,1000,input_dim)
    hparams['resblock_hparams']['wr']         = 1e-6
    hparams['resblock_hparams']['max_epoch']  = 20
    hparams['resblock_hparams']['early_stop'] = 10
    hparams['fg_eta']    = 1e-3
    hparams['max_iters'] = 50

model = ResFGB( **hparams )
model.fit( X, Y )

train_loss, train_acc = model.evaluate( X, Y )

logging.info( '- Result -' )
logging.info( 'train_loss: {0:5.4f}, train_acc: {1:4.3f}'\
              .format( train_loss, train_acc ) )

test_loss, test_acc  = model.evaluate( Xt,  Yt )
logging.info( 'test_loss : {0:5.4f}, test_acc : {1:4.3f}'\
              .format( test_loss, test_acc ) )

