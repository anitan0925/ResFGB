# coding : utf-8

"""
Get default hyper-parameters of ResFGB.
"""

from __future__ import print_function, absolute_import, division, unicode_literals


def get_hyperparams(n_data, input_dim, n_class):
    model_hparams = {'shape': (input_dim, n_class),
                     'bias': True,
                     'wr': 1 / n_data,
                     'eta': 1e-2,
                     'momentum': 0.9,
                     'minibatch_size': 100,
                     'scale': 1.0,
                     'max_epoch': 100,
                     'tune_eta': True,
                     'eval_iters': 1000,
                     'early_stop': 10}

    resblock_hparams = {'shape': (input_dim, 100, 100, 100, 100, input_dim),
                        'wr': 1 / n_data,
                        'eta': 1e-2,
                        'momentum': 0.9,
                        'minibatch_size': 100,
                        'scale': 1.0,
                        'max_epoch': 50,
                        'tune_eta': True,
                        'eval_iters': 1000,
                        'early_stop': 10}

    hparams = {'model_type': 'logistic',
               'model_hparams': model_hparams,
               'resblock_hparams': resblock_hparams,
               'fg_eta': 1e-1,
               'max_iters': 30,
               'seed': 1}

    return hparams
