#encoding: utf8

import numpy as np
import sys
import pdb

'''
to resolve Ax = b.
first LU,then solve it
only for square matrix, and det(A) != 0
'''

from lu import LU

def solve(A, b):
    epsilon = 1e-9
    A_shape = A.shape
    A_ext = np.vstack((A.T,b)).T

    L,U = LU(A_ext)

    x = np.zeros(A_shape[0])
    x[-1] = U[-1, -1]
    for i in range(A_shape[0] - 2, -1, -1):
        x[i] = U[i, -1] - np.dot(U[i, (i+1):-1], x[(i+1):])
        
    return x

A = np.array([[1,2,3,2],
              [1,2,7,5],
              [4,9,2,6],
              [6,1,5,8]]);

A = np.array([[1,2,3,2,1],
              [1,2,7,5,2],
              [4,9,2,6,3],
              [6,1,5,8,4]]);

A=np.array([[1,2,3], 
           [4,5,6], 
           [1,1,5]], dtype='float');

A = np.array([[-4,-2,4],
              [-2,-1,2],
              [4,2,-4]]);

b = np.array([1., 2., 3.])
b = np.zeros(3);
x_base = np.array([ 0.33333333, -0.66666667,  0.66666667])

if __name__=='__main__':
    x = solve(A, b)

    print "A:"
    print A
    print "\nx:"
    print x 
    print "\nx_base:"
    print x_base 
    print "\n(x - x_base)/x_base:"
    print np.sum(np.abs(x-x_base)) / np.sum(np.abs(x_base))
    print "\nb:"
    print b 
    print "\nA*x:"
    print np.dot(A,x)

