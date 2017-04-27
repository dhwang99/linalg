#encoding: utf8

import numpy as np
import sys
import pdb

'''
利用LU分解求逆

用高斯消元法求逆。E1*E2*E3*A = I

'''

def inv(A, retU=False):
    A_shape = A.shape
    E = np.eye(A_shape[0], A_shape[0])  # 左乘伴随矩阵
    A_ext = np.hstack((A,E))          # 生成增广矩阵

    #LU first
    for i in range(A_shape[0] - 1):
        for j in range(i+1, A_shape[0]):
            t = 1.0 * A_ext[j,i]/A_ext[i,i]
            A_ext[j,:] = A_ext[j,:] -  t * A_ext[i,:];

    #standard
    for i in range(A_shape[0]):
        A_ext[i,:] = A_ext[i,:] * (1.0/A_ext[i,i]) #the main element's value equal to 1.

    #把U矩阵转为对角矩阵
    for i in range(A_shape[0] - 1, -1, -1):
        for j in range(i+1, A_shape[0]):
            A_ext[i,:] -= A_ext[j,:] * A_ext[i,j]

    U = A_ext[:, 0:A_shape[1]]
    L_inv = A_ext[:, A_shape[1]:]
    
    if retU:
        return (L_inv, U)
    else:
        return L_inv

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

A=np.array([[1,2,3], 
           [4,5,6], 
           [1,1,5]], dtype='float');


if __name__=='__main__':
    L_inv,U = inv(A, retU=True)

    print "A:"
    print A

    print "L_inv:\n", L_inv
    print "U:\n", U

    print "L*U:\n", np.dot(np.linalg.inv(L_inv), U)

    print "L_Inv * A:\n", np.dot(L_inv, A)
