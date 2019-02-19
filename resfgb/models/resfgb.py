# coding : utf-8

"""
ResFGB for multiclass classificcation problems.
"""

from __future__ import print_function, absolute_import, division, unicode_literals
from logging import getLogger, ERROR
import time
import sys
import numpy as np
import theano
from resfgb.models import LogReg, SVM, ResGrad

logger = getLogger(__name__)


class ResFGB(object):
    def __init__(self, model_type='logistic', model_hparams={}, resblock_hparams={},
                 fg_eta=None, max_iters=10, seed=99, proc_batch_size=10000):

        self.show_param(model_type,
                        model_hparams['tune_eta'],
                        model_hparams['max_epoch'],
                        model_hparams['early_stop'],
                        max_iters)

        self.__tune_eta__ = model_hparams['tune_eta']
        self.__max_epoch__ = model_hparams['max_epoch']
        self.__early_stop__ = model_hparams['early_stop']
        del model_hparams['tune_eta']
        del model_hparams['max_epoch']
        del model_hparams['early_stop']

        if model_type == 'logistic':
            self.__model__ = LogReg(seed=seed, **model_hparams)
        elif model_type == 'smooth_hinge':
            self.__model__ = SVM(seed=seed, **model_hparams)
        else:
            logger.log(ERROR, 'invalid model_type: {0}'.format(model_type))
            sys.exit(-1)

        self.__max_iters__ = max_iters
        self.__fg__ = ResGrad(self.__model__, eta=fg_eta,
                              resblock_hparams=resblock_hparams,
                              seed=seed, proc_batch_size=proc_batch_size)

    def show_param(self, model_type, tune_eta, max_epoch, early_stop, max_iters):
        logger.info('{0:<5}{1:^26}{2:>5}'.format('-' * 5, 'ResFGB setting', '-' * 5))
        logger.info('{0:<15}{1:>21}'.format('model_type', model_type))
        logger.info('{0:<15}{1:>21}'.format('tune_eta', tune_eta))
        logger.info('{0:<15}{1:>21}'.format('max_epoch', max_epoch))
        logger.info('{0:<15}{1:>21}'.format('early_stop', early_stop))
        logger.info('{0:<15}{1:>21}'.format('max_iters', max_iters))

    def evaluate(self, Z, Y, sample_f=True):
        if sample_f:
            Z = self.__fg__.predict(Z, Z)
            loss, acc = self.__model__.evaluate(Z, Y)
        else:
            loss, acc = self.__model__.evaluate(Z, Y)

        return loss, acc

    def fit(self, X, Y, Xv=None, Yv=None, use_best_iter=False):

        logger.info('{0:<5}{1:^26}{2:>5}'.format('-' * 5, 'Training ResFGB', '-' * 5))

        best_val_acc = 0.
        best_val_loss = 1e+10
        best_param = None
        best_n_layers = None
        total_time = 0.

        Z = np.array(X)

        if Xv is not None:
            monitor = True
            Zv = np.array(Xv)
        else:
            monitor = False
            Zv = None

        for n_iter in range(self.__max_iters__):
            #----- apply functional gradient -----
            stime = time.time()
            if n_iter >= 1:
                Z = self.__fg__.apply(X, Z, lfrom=n_iter - 1)
                if monitor:
                    Zv = self.__fg__.apply(Xv, Zv, lfrom=n_iter - 1)

            #----- fit and evaluate -----
            self.__model__.optimizer.reset_func()
            if self.__tune_eta__ and (n_iter == 0):
                self.__model__.determine_eta(Z, Y)

            self.__model__.fit(Z, Y, self.__max_epoch__, early_stop=self.__early_stop__)

            etime = time.time()
            total_time += etime - stime

            train_loss, train_acc = self.evaluate(Z, Y, sample_f=False)
            logger.info('layer: {0:4}, time:{1:>14.1f} sec'
                        .format(n_iter, total_time))
            logger.info('train_loss: {0:5.4f}, train_acc: {1:4.3f}'
                        .format(train_loss, train_acc))

            if monitor:
                val_loss, val_acc = self.evaluate(Zv, Yv, sample_f=False)
                logger.info('val_loss: {0:8.4f}, val_acc: {1:7.3f}'
                            .format(val_loss, val_acc))

                if val_acc > best_val_acc:
                    best_n_layers = n_iter
                    best_val_acc = val_acc
                    best_val_loss = val_loss
                    best_param = self.__model__.get_params(real_f=True)

            #----- compute weight matrix -----
            stime = time.time()
            if False: #subsample
                indices = range(Z.shape[0])
                np.random.shuffle(indices)
                n_samples = int( 0.5 * len(indices) )
                self.__fg__.compute_weight(X[indices[:n_samples]],
                                           Z[indices[:n_samples]],
                                           Y[indices[:n_samples]])
            else:
                self.__fg__.compute_weight(X, Z, Y)
            etime = time.time()
            total_time += etime - stime

        #----- apply functional gradient -----
        stime = time.time()
        if self.__max_iters__ >= 1:
            Z = self.__fg__.apply(X, Z, lfrom=self.__max_iters__ - 1)
            if monitor:
                Zv = self.__fg__.apply(Xv, Zv, lfrom=self.__max_iters__ - 1)

        #----- fit and evaluate -----
        self.__model__.optimizer.reset_func()
        self.__model__.fit(Z, Y, self.__max_epoch__, early_stop=self.__early_stop__)

        etime = time.time()
        total_time += etime - stime

        train_loss, train_acc = self.evaluate(Z, Y, sample_f=False)
        logger.info('layer: {0:4}, time:{1:>14.1f} sec'
                    .format(self.__max_iters__, total_time))
        logger.info('train_loss: {0:5.4f}, train_acc: {1:4.3f}'
                    .format(train_loss, train_acc))

        if monitor:
            val_loss, val_acc = self.evaluate(Zv, Yv, sample_f=False)
            logger.info('val_loss: {0:8.4f}, val_acc: {1:7.3f}'
                        .format(val_loss, val_acc))

            if val_acc > best_val_acc:
                best_n_layers = self.__max_iters__
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_param = self.__model__.get_params(real_f=True)

        #----- finalize -----
        if monitor and use_best_iter is True:
            if best_n_layers < self.__max_iters__:
                del self.__fg__.params[best_n_layers:]
                self.__model__.set_params(best_param)

        if monitor:
            if use_best_iter is True:
                return (best_n_layers, best_val_loss, best_val_acc)
            else:
                return (self.__max_iters__, val_loss, val_acc)
        else:
            return (None, None, None)
