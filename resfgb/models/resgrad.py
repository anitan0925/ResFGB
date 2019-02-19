# coding : utf-8

"""
Learning procedure of residual blocks in ResFGB.
"""

from __future__ import print_function, absolute_import, division, unicode_literals
import sys
from logging import getLogger, DEBUG
import numpy as np
import theano
import theano.tensor as T
from resfgb.utils import minibatches, minibatches2, minibatch_indices
from resfgb.models import layers as L
from resfgb.models.mlp_block import MLPBlock, MLPBlock2

logger = getLogger(__name__)

try:
    from numba import jit, f4

    @jit(f4[:, :](f4[:, :], f4[:, :]), nopython=True)
    def __dot__(a, b):
        return np.dot(a, b)
except ImportError:
    logger.warning('fail to use Numba matrix product')

    def __dot__(a, b):
        return np.dot(a, b)


class ResGrad(object):
    def __init__(self, model, eta, resblock_hparams={},
                 seed=99, proc_batch_size=10000):

        self.__FW__ = True

        self.show_param(eta,
                        resblock_hparams['tune_eta'],
                        resblock_hparams['max_epoch'],                       
                        resblock_hparams['early_stop'],
                        seed)

        self.params = []
        self.__eta__ = eta
        self.__tune_eta__ = resblock_hparams['tune_eta']
        self.__max_epoch__ = resblock_hparams['max_epoch']
        self.__early_stop__ = resblock_hparams['early_stop']
        self.__momentum__ = resblock_hparams['resblock_momentum']
        if self.__momentum__ > 0:
            self.__nesterov__ = True
        else:
            self.__nesterov__ = False
        self.__velocity__ = None
        
        del resblock_hparams['tune_eta']
        del resblock_hparams['max_epoch']
        del resblock_hparams['early_stop']
        del resblock_hparams['resblock_momentum']

        """
        Large-data will be divided and processed by the batch_size just for the computational efficiency.
        If you encounter the segmentation fault, let you reduce the batch_size.
        However, note that it may slow down the computation time for small/middle-data sets.
        """
        self.__batch_size__ = proc_batch_size

        self.__regressor_params__ = []
        self.__current_itr__ = -1

        # compile
        # self.__regressor__ = MLPBlock(seed=seed, **resblock_hparams)
        self.__regressor__ = MLPBlock2(seed=seed, **resblock_hparams)
        self.__model__ = model
        self.__zgrad__ = T.grad(cost=self.__model__.loss, wrt=self.__model__.Z)
        self.__zgrad_func__ = theano.function([self.__model__.Z, self.__model__.Y],
                                              self.__zgrad__)

    def show_param(self, eta, tune_eta, max_epoch, early_stop, seed):
        logger.info('{0:<5}{1:^26}{2:>5}'.format('-' * 5, 'ResGrad setting', '-' * 5))
        logger.info('{0:<15}{1:>21.7f}'.format('fg_eta', eta))
        logger.info('{0:<15}{1:>21}'.format('tune_eta', tune_eta))
        logger.info('{0:<15}{1:>21}'.format('max_epoch', max_epoch))
        logger.info('{0:<15}{1:>21}'.format('early_stop', early_stop))
        logger.info('{0:<15}{1:>21}'.format('seed', seed))

    def predict(self, X, Z, wr):
        return self.apply(X, Z, wr)

    def set_regressor_params(self, l):
        if self.__current_itr__ == l or l < 0:
            return
        else:
            self.__current_itr__ = l
            self.__regressor__.set_params(self.__regressor_params__[l])

    def solve_gradient(self, X, Z, zgrads, n_layers):
        self.set_regressor_params(n_layers - 1)
        self.__regressor__.optimizer.reset_func()

        eps = 1e-10
        znorm = np.sqrt(np.mean(zgrads ** 2, axis=1)) + eps
        logger.log(DEBUG, 'min: {0:12.5f}, max: {1:12.5f}'
                   .format(np.min(znorm), np.max(znorm)))

        if self.__tune_eta__ and (n_layers == 0):
            self.__regressor__.determine_eta(X, Z, zgrads / znorm[:, None])

        self.__regressor__.fit(X, Z, zgrads / znorm[:, None], self.__max_epoch__,
                               early_stop=self.__early_stop__)

        self.__current_itr__ = len(self.__regressor_params__)
        self.__regressor_params__.append(self.__regressor__.get_params(real_f=True))

    def approximate_gradient(self, X, Z, l):
        self.set_regressor_params(l)
        return self.__regressor__.predict(X,Z)

    def compute_weight(self, X, Z, Y):
        """
        Compute the weight matrix for functional gradient method.

        Arguments
        ---------
        Z : Numpy array. 
            Represents data.
        Y : Numpy array.
            Represents label.
        """
        n, d = Z.shape
        n_layers = len(self.params)

        fullbatch_mode = True if n <= self.__batch_size__ or self.__batch_size__ is None else False

        if fullbatch_mode:
            # (n, d)
            zgrads = self.__zgrad_func__(Z, Y)
            fgnorm = np.sqrt(np.mean( np.sum(zgrads**2, axis=1) ))            
            if self.__FW__:
                eta = 1./fgnorm
            else:
                eta = self.__eta__ * fgnorm
            
            self.solve_gradient(X, Z, zgrads, n_layers)
            Zl = self.approximate_gradient(X, Z, n_layers)

            eta = self.__current_itr__ + 1

            if self.__nesterov__ and self.__velocity__ is None:
                self.__velocity__ = np.zeros( shape=Z.shape ) 

            if self.__nesterov__:
                self.__velocity__ = self.__momentum__ * self.__velocity__ \
                                    + eta * zgrads

                Wl = __dot__(Zl.T, eta * zgrads
                             + self.__momentum__ * self.__velocity__)
            else:
                # Zl: (n,emb_dim), zgrads: (n,d)
                # Wl: (emb_dim,d)
                Wl = __dot__(Zl.T, eta * zgrads)
        else:
            Wl = np.zeros(shape=(d, d), dtype=theano.config.floatX)
            zgrads = []
            for i, start in enumerate(range(0, n, self.__batch_size__)):
                end = min(n, start + self.__batch_size__)
                Zb = Z[start:end]
                Yb = Y[start:end]
                b = Zb.shape[0]
                zgrads.append(self.__zgrad_func__(Zb, Yb) * float(b) / float(n))

            zgrads = np.vstack(zgrads)
            fgnorm = np.sqrt(np.mean( np.sum(zgrads**2, axis=1) ))            
            if self.__FW__:
                eta = 1./fgnorm
            else:
                eta = self.__eta__ * fgnorm            
            
            self.solve_gradient(X, Z, zgrads, n_layers)

            for i, start in enumerate(range(0, n, self.__batch_size__)):
                end = min(n, start + self.__batch_size__)
                Xb = X[start:end]
                Zb = Z[start:end]
                Yb = Y[start:end]
                zgradb = zgrads[start:end]
                Zlb = self.approximate_gradient(Xb, Zb, n_layers)

                if self.__nesterov__:
                    if self.__velocity__ is None:
                        self.__velocity__ = [ np.zeros( shape=Zb.shape ) ]
                    elif i+1 > len( self.__velocity__ ):
                        self.__velocity__.append( np.zeros( shape=Zb.shape ) )
             
                if self.__nesterov__:
                    self.__velocity__[i] = self.__momentum__ * self.__velocity__[i] \
                                           + eta * zgradb
                    # (emb_dim,d)
                    Wl += __dot__(Zlb.T, eta * zgradb
                                   + self.__momentum__ * self.__velocity__[i])
                else:
                    Wl += __dot__(Zlb.T, eta * zgradb)

        self.params.append(Wl)
        # print("eta: %f, fgnorm: %f" % (eta, fgnorm))
        
    def apply(self, X, Z_, wr, lfrom=0):
        """
         Perform functional gradient descent.
        """

        if len(self.params) == 0:
            return Z_

        n = Z_.shape[0]
        fullbatch_mode = True if n <= self.__batch_size__ else False
        shape = Z_.shape

        if fullbatch_mode:
            Z = np.array(Z_)
            if wr>0:
                Z *= 1. - wr
                
            for i, Wl in enumerate(self.params[lfrom:]):
                l = lfrom + i
                Zk = self.approximate_gradient(X, Z, l)
                Tk = __dot__(Zk, Wl)
                if self.__FW__:
                    scale = 2./(self.__eta__+l)
                    Tk *= scale
                    Z *= 1.-scale
                Z -= Tk

            return Z
        else:
            Z = []
            for start in range(0, n, self.__batch_size__):
                end = min(n, start + self.__batch_size__)
                Xb = X[start:end]
                Zb = np.array(Z_[start:end])
                if wr>0:
                    Zb *= 1. - wr
                shape = Zb.shape

                for i, Wl in enumerate(self.params[lfrom:]):
                    l = lfrom + i
                    Zkb = self.approximate_gradient(Xb, Zb, l)
                    Tkb = __dot__(Zkb, Wl)
                    if self.__FW__:
                        scale = 2./(self.__eta__+l)
                        Tkb *= scale
                        Zb *= 1.-scale                        
                    Zb -= Tkb

                Z.append(Zb)
            return np.vstack(Z)
