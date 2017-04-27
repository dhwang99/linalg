#encoding: utf8

import numpy as np
import sys
import pdb

'''
LU分解
用 高斯消元法来做LU分解

对  Ax = b, 当A固定，需要经常解x时，用这种方法比较合适。如下方法：
Ly = b, L为下三角矩阵,n**2/2 次计算 得到 y
Ux = y, U为上三角矩阵, n**2/2次计算得到 x
'''

from inv import inv

def LU(A):
    A_shape = A.shape
    E = np.eye(A_shape[0], A_shape[0])  # 左乘伴随矩阵
    A_ext = np.hstack((A,E))          # 生成增广矩阵

    #LU first
    for i in range(A_shape[0] - 1):
        #pdb.set_trace()
        for j in range(i+1, A_shape[0]):
            t = 1.0 * A_ext[j,i]/A_ext[i,i]
            A_ext[j,:] = A_ext[j,:] -  t * A_ext[i,:];

    #standard
    for i in range(A_shape[0]):
        A_ext[i,:] = A_ext[i,:] * (1.0/A_ext[i,i]) #the main element's value equal to 1.

    U = A_ext[:, 0:A_shape[1]]
    L_inv = A_ext[:, A_shape[1]:]
    L = inv(L_inv)
        
    return (L,U)

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

#特征值为 7,7,-2
A = np.array([[3,-2,4],
              [-2,6,2],
              [4,2,3]]);

A=np.array([[1,2,3, 1], 
           [4,5,6, 1], 
           [1,1,5, 7]], dtype='float');



if __name__=='__main__':
    L,U = LU(A)

    print "A:"
    print A

    print "L:\n", L
    print "U:\n", U

    print "L*U:\n", np.dot(L, U)
