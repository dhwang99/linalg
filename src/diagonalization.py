# encoding: utf8

import numpy as np
import sys
import pdb

'''
对角化。借用 qr分解.
仅当A为对称阵时，Q列向量正交
注意：只有方阵才能对角化
'''

from eig_by_qr import eig
from inv import inv

def  diagonalization(A):
    #1. 求特征值和特征向量
    e,v = eig(A1)
    P = v
    D = np.diag(e)
    P_inv = inv(P)

    return (P,D,P_inv)

if __name__=='__main__':
    A = np.array([[3,-2,4],
                  [-2,6,2],
                  [4,2,3]]);
    
    A1 = A
    
    A = np.array([[1,2,3,2],
                  [1,2,7,5],
                  [4,9,2,6],
                  [6,1,5,8]]);
    
    A1 = np.dot(A.T, A)
    
    P,D,P_inv = diagonalization(A1)
    print "A1:"
    print A1
    print "P*D*inv(P):"
    print np.dot(np.dot(P, D), P_inv)
