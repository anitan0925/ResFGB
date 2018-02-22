# config : utf-8

from __future__ import print_function, absolute_import, division, unicode_literals
import sys, os
from . import binarize_libsvm_data 
from . import data_loader
if sys.version_info[0] == 2:
    import cPickle
else:
    import _pickle as cPickle

def save( train_url, train_file, test_url=None, test_file=None ):
    test_f = True if test_url is not None and test_file is not None else False
    bz2_f = True if os.path.splitext(train_url)[1] == '.bz2' else False

    train_raw = os.path.basename(train_url)
    if not os.path.exists( train_raw ):
        os.system( 'wget %s' % train_url )
    if test_f:
        test_raw = os.path.basename(test_url)
        if not os.path.exists( test_raw ):
            os.system( 'wget %s' % test_url )
        
    X_, Y_, xdic, ydic = binarize_libsvm_data.preprocess( train_raw, bz2_f=bz2_f )
    if test_f:
        Xt_, Yt_, xdic, ydic = binarize_libsvm_data.preprocess( test_raw, bz2_f=bz2_f )
    else:
        Xt_, Yt_ = None, None

    X, Y = binarize_libsvm_data.binarize( X_, Y_, xdic, ydic )
    fout = open( train_file, 'wb' )
    cPickle.dump( [X,Y], fout )
    fout.close()

    if test_f:
        Xt, Yt = binarize_libsvm_data.binarize( Xt_, Yt_, xdic, ydic )
        fout = open( test_file, 'wb' )
        cPickle.dump( [Xt,Yt], fout )
        fout.close()

def get_ijcnn1():
    train_file = './data/ijcnn1.data'
    test_file = './data/ijcnn1.t.data'
    split = True
    standardize = True
    normalize   = False
    bias        = True

    if not os.path.exists( train_file ) or not os.path.exists( test_file ):
        train_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2'
        test_url  = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2'
        save( train_url, train_file, test_url, test_file )
        
    X, Y, Xt, Yt = data_loader.load( train_file, test_file, split, 
                                     standardize, normalize, bias )
    return X, Y, Xt, Yt
        
