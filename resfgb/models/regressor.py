# coding : utf-8

from __future__ import print_function, absolute_import, division, unicode_literals
import logging, time
import numpy as np
from resfgb.utils import minibatches
from resfgb.models.model import Model

class Regressor(Model):
    def __init__( self, eta, scale, minibatch_size, eval_iters, seed, log_level ):
        """
        eta            : float.
        scale          : float
        minibatch_size : integer.
                         Minibatch size to calcurate stochastic gradient.
        eval_iters     : Integer.
                         The number of iterations for evaluating eta.
        seed           : integer.
                         Seed for random module.
        """
        super( Regressor, self ).__init__( seed )

        self.__eta          = eta
        self.__scale        = scale
        self.minibatch_size = minibatch_size
        self.eval_iters     = eval_iters
        self.log_level      = log_level
    
    def evaluate( self, Z, Y ):
        n, loss = 0, 0.
        minibatch_size = np.min( (10000, Z.shape[0]) )
        for (Zb,Yb) in minibatches( minibatch_size, Z, Y, shuffle=False ):
            n_ = Zb.shape[0]
            loss_ = self.loss_func(Zb,Yb)
            loss  = (n/(n+n_))*loss + (n_/(n+n_))*loss_
            n    += n_

        return loss + self.reg_func()

    def __save_param( self ):
        self.__saved_params = self.get_params( real_f=True )

    def __load_param( self ):
        self.set_params( self.__saved_params, real_f=True )

    def evaluate_eta( self, X, Y, eta ):
        self.__save_param()
        self.optimizer.set_eta(eta)

        n_iters = 0
        eval_f = True
        while eval_f:
            for (Xb,Yb) in minibatches( self.minibatch_size, X, Y, shuffle=True ):
                if n_iters >= self.eval_iters:
                    eval_f = False
                    break
                self.optimizer.update_func(Xb,Yb)
                n_iters += 1

        val = self.evaluate(X,Y)
        self.__load_param()
        self.optimizer.reset_func()

        return val

    def determine_eta( self, X, Y, factor=2. ):
        val0     = self.evaluate( X, Y )

        low_eta  = self.__eta
        low_val  = self.evaluate_eta( X, Y, low_eta )
        low_val  = np.inf if np.isnan( low_val ) else low_val

        high_eta = factor * low_eta
        high_val = self.evaluate_eta( X, Y, high_eta )
        high_val = np.inf if np.isnan( high_val ) else high_val

        decrease_f = True if ( np.isinf(low_val) 
                               or low_val < high_val 
                               or val0 < low_val 
                               or val0 < high_val ) else False

        if decrease_f:
            while low_val < high_val or np.isinf(low_val) or val0 < low_val:
                high_eta = low_eta
                high_val = low_val
                low_eta  = high_eta / factor
                low_val  = self.evaluate_eta( X, Y, low_eta )
                low_val  = np.inf if np.isnan( low_val ) else low_val
                self.__eta = high_eta 
        else:
            while low_val > high_val:
                low_eta  = high_eta
                low_val  = high_val
                high_eta = low_eta * factor
                high_val = self.evaluate_eta( X, Y, high_eta )
                high_val = np.inf if np.isnan( high_val ) else high_val
                self.__eta = low_eta 

        self.__eta *= self.__scale
        self.optimizer.set_eta( self.__eta )

        logging.info( 'new_eta (regressor): {0:>15.7f}'.format( self.__eta ) )

    def fit( self, X, Y, max_epoch, early_stop=-1 ):
        """
        Run algorigthm for up to (max_epoch) on training data X.
        
        Arguments
        ---------
        optimizer  : Instance of optimizer class.
        X          : Numpy array. 
                     Training data.
        Y          : Numpy array.
                     Training label.
        max_epoch  : Integer.
        early_stop : Integer.
        """

        logging.log( self.log_level, 
                     '{0:<5}{1:^26}{2:>5}'.format( '-'*5, 'Training regressor', '-'*5 ) )

        total_time = 0.

        self.__save_param()
        success = False

        init_train_loss = self.evaluate( X, Y )
        best_loss  = 1e+10
        best_epoch = 0

        while success is False:
            success = True

            for e in range(max_epoch):
                stime = time.time()
                for (Xb,Yb) in minibatches( self.minibatch_size,
                                            X, Y, shuffle=True ):
                    self.optimizer.update_func(Xb,Yb)
                etime = time.time()
                total_time += etime-stime

                train_loss = self.evaluate( X, Y )
                if np.isnan(train_loss) or np.isinf(train_loss) \
                   or (2*init_train_loss + 1) <= train_loss:
                    eta = self.optimizer.get_eta() / 2.
                    self.optimizer.set_eta( eta )
                    success = False
                    self.__load_param()
                    self.optimizer.reset_func()
                    logging.log( self.log_level, 'the learning process diverged' )
                    logging.log( self.log_level, 'retrain a model with a smaller learning rate: {0}'\
                                .format( eta ) )
                    break

                logging.log( self.log_level, 'epoch: {0:4}, time: {1:>13.1f} sec'\
                             .format( e, total_time ) )
                logging.log( self.log_level, 'train_loss: {0:5.4f}'.format( train_loss ) )

                # early_stopping 
                if train_loss < 0.999*best_loss:
                    best_loss = train_loss
                    best_epoch = e

                if early_stop > 0 and e - best_epoch >= early_stop:
                    success = True
                    break

        return None
