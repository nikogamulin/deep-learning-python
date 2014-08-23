'''
Created on 22. avg. 2014

@author: niko
'''

import cPickle, gzip, numpy
import theano
import theano.tensor as T

def shared_dataset(data_xy):
    data_x, data_y = data_xy
    shared_x = theano.shared(numpy.asarray(data_x, dtype=theano.config.floatX))
    shared_y = theano.shared(numpy.asarray(data_y, dtype=theano.config.floatX))
    
    return shared_x, T.cast(shared_y, 'int32')

#load dataset
f = gzip.open('./data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

test_set_x, test_set_y = shared_dataset(test_set)

batch_size = 500