# coding : utf-8

from __future__ import print_function, absolute_import, division, unicode_literals
import sys
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.signal import pool

relu_alpha = 0.


def uniform_param(shape, scale=5e-2):
    return theano.shared(
        np.random.uniform(size=shape, low=-scale, high=scale).astype(
            dtype=theano.config.floatX))


def zeros_param(shape):
    return theano.shared(np.zeros(shape, dtype=theano.config.floatX))


def ones_param(shape):
    return theano.shared(np.ones(shape, dtype=theano.config.floatX))


def linear_param(input_dim, output_dim, scale=None):
    if scale is None:
        scale = np.sqrt(2. / float(input_dim + output_dim))
        scale *= np.sqrt(3)

    shape = (input_dim, output_dim)
    return uniform_param(shape, scale=scale)


def conv_param(output_dim, input_dim, ksize, stride=1, scale=None):
    fan_in = input_dim * ksize**2
    fan_out = output_dim * ksize**2 / (stride**2)

    if scale is None:
        scale = np.sqrt(4. / (fan_in + fan_out))
        scale *= np.sqrt(3)

    shape = (output_dim, input_dim, ksize, ksize)

    return uniform_param(shape, scale=scale)


def deconv_param(output_dim, input_dim, ksize, stride=2, scale=None, name=''):
    fan_in = input_dim * ksize**2 / (stride**2)
    fan_out = output_dim * ksize**2

    if scale is None:
        scale = np.sqrt(4. / (fan_in + fan_out))
        scale *= np.sqrt(3)

    shape = (output_dim, input_dim, ksize, ksize)

    return uniform_param(shape, scale=scale, name=name)


def Act(X, act_type, real_f=False):
    global relu_alpha

    if real_f:
        return Act_real(X, act_type)

    if act_type == 'sigmoid':
        __f = T.nnet.sigmoid
    elif act_type == 'tanh':
        __f = T.tanh
    elif act_type == 'softmax':
        __f = T.nnet.softmax
    elif act_type == 'relu':
        def __f(X): return T.nnet.relu(X, relu_alpha)
    elif act_type == 'softplus':
        __f = T.nnet.softplus

    return __f(X)


def Act_real(X, act_type):
    global relu_alpha

    if act_type == 'sigmoid':
        exit(-1)
    elif act_type == 'tanh':
        __f = np.tanh
    elif act_type == 'softmax':
        __f = softmax
    elif act_type == 'relu':
        def __f(X): return relu(X, relu_alpha)
    elif act_type == u'softplus':
        exit(-1)

    return __f(X)


def softmax(X):
    n = X.shape[0]
    maxes = np.max(X, axis=1).reshape(n, 1)
    v = np.exp(X - maxes)
    z = np.sum(v, axis=1).reshape(n, 1)
    return v / z


def FullConnect(X, params):
    if len(params) == 2:
        b, W = params
        output = T.dot(X, W) + b
    else:
        W = params[0]
        output = T.dot(X, W)
    return output


def Conv2d(X, W, b=None, subsample=(1, 1), border_mode=u'half'):
    if b is not None:
        # dimshuffle(u'x',0,'x','x') -> (1,b_shape,1,1)
        output = T.nnet.conv2d(X, W, subsample=subsample,
                               border_mode=border_mode) \
            + b.dimshuffle(u'x', 0, 'x', 'x')
    else:
        output = T.nnet.conv2d(X, W, subsample=subsample,
                               border_mode=border_mode)
    return output


def Pool2d(X, k_size, ignore_border=False, mode='max'):
    output = pool.pool_2d(X, ws=(k_size, k_size), ignore_border=ignore_border, mode=mode)

    return output


def relu(X, alpha):
    return np.maximum(X, alpha * X)


def FullConnect_nd(X, params):
    if len(params) == 2:
        b, W = params
        output = nd.dot(X, W) + b
    else:
        W = params[0]
        output = nd.dot(X, W)
    return output


def Loss(X, Y, ltype=u'cross_entropy'):
    if ltype == u'cross_entropy':
        loss = -T.mean(T.log(X)[T.arange(Y.shape[0]), Y])
        return loss
    elif ltype == u'squared_error':
        loss = T.mean((X - Y)**2)
        return loss
    elif ltype == u'huber':
        delta = 1.
        diff = X - Y
        a = 0.5 * (diff**2)
        b = delta * (abs(diff) - delta / 2.)
        loss = T.mean(T.switch(abs(diff) <= delta, a, b))
        return loss
    else:
        sys.exit(-1)


def normalize(X, eps=1e-0):
    return X / (T.sqrt(T.sum(X**2, axis=1)) + eps)[:, None]


def Flatten(X):
    return X.flatten(2)


def Flatten_real(X):
    n, c, d1, d2 = X.shape
    return X.reshape((n, c * d1 * d2))


def Batchnorm(x, b, g, ave_m, ave_v, train_f=True, eps=1e-4, decay=0.9):
    # b: offset, g: scale
    if x.ndim == 4:
        if train_f:
            m_ = T.mean(x, axis=[0, 2, 3])
            m = m_.dimshuffle('x', 0, 'x', 'x')
            v_ = T.mean(T.sqr(x - m), axis=[0, 2, 3])
            v = v_.dimshuffle('x', 0, 'x', 'x')

            n = T.cast(T.prod(x.shape) / T.prod(m.shape), u'float32')

            updates = [(ave_m, decay * ave_m + (1. - decay) * m_),
                       (ave_v, decay * ave_v + (n / (n - 1.)) * (1. - decay) * v_)]
        else:
            m = ave_m.dimshuffle('x', 0, 'x', 'x')
            v = ave_v.dimshuffle('x', 0, 'x', 'x')
            updates = []

        x = (x - m) / T.sqrt(v + eps)
        x = x * g.dimshuffle('x', 0, 'x', 'x') + b.dimshuffle('x', 0, 'x', 'x')

    elif x.ndim == 2:
        if train_f:
            m = T.mean(x, axis=0)
            v = T.mean(T.sqr(x - m), axis=0)

            n = T.cast(T.prod(x.shape) / T.prod(m.shape), u'float32')

            updates = [(ave_m, decay * ave_m + (1. - decay) * m),
                       (ave_v, decay * ave_v + (n / (n - 1.)) * (1. - decay) * v)]
        else:
            m = ave_m
            v = ave_v
            updates = []

        x = (x - m) / T.sqrt(v + eps)
        x = x * g + b

    return x, updates
