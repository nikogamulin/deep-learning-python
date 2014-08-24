'''
Created on Aug 23, 2014

@author: GamulinN
'''

import theano.tensor as T
from theano import function


if __name__ == '__main__':
    x = T.scalar('x')
    y = T.scalar('y')
    z = x + y
    f = function([x,y], z)
    f(2,3)