#encoding: utf8
'''
'''

import numpy as np
import sys
import pdb

from eig_by_qr import eig, cal_nullspace 

'''
                             +   +
SVD分解: A=USV', 或简化版 A=U S V' , U: m*r, S: r*r, V: n*r
 +   +   +-1    + 
A = Vnr Srr  U'rm, 伪逆

对A' * A而言，特征值恒>=0:
   ||Ax||**2 = x'A'Ax = x'*lambda * x = lambda * x' * x = lambda * ||x||**2 
   故而 lambda >= 0

'''

'''
retSimple: 是否返回简化svd分解
'''

def svd(A, retSimple=False):
    epsilon = 1e-4
    A_shape = A.shape
    ATA = np.dot(A.T, A)
    #pdb.set_trace()
    e,v = eig(ATA)
    e_sqrt = np.sqrt(np.take(e, np.where(e > epsilon)[0]))
    r = len(e_sqrt)

    U = np.zeros((A_shape[0], A_shape[0]))  #m*m
    S = np.zeros(A_shape)                   #m*n
    V = np.zeros((A_shape[1], A_shape[1]))  #n*n

    for i in range(r):
        ui = np.dot(A, v[:,i])
        ui = ui / np.sqrt(np.dot(ui, ui))
        U[:,i] = ui

        S[i,i] = e_sqrt[i]

    #pdb.set_trace()
    if retSimple:
        U = U[:, :r]
        S = S[:r,:r]
        V = v[:, :r].T
    else:
        if r < A_shape[0]:
            ns = cal_nullspace(U.T)  #求U.T的零空间, 即与U[0:r]正交的空间
            U[:,r:] = ns
        V = v.T

    return (U,S,V)

if __name__=='__main__':
    A=np.array([[4,11,14],[8,7,-2]])

    A=np.array([[1,-1], [-2,2], [2,-2]])

    U,S,V = svd(A, retSimple = True)

    print  "U:" 
    print U
    print  "S:" 
    print S
    print  "V:" 
    print V

    print "A:"
    print A


    print "USV:"
    print np.dot(np.dot(U,S), V) 

    u,s,v = np.linalg.svd(A)


