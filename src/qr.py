#encoding: utf8
'''
Gram-Schmidt正交化
其它方法(如Householder变换, Givens旋转 等)， 后继也考虑实现一下


注意下， QR分解并不需要A是方阵，它只是求了一个A子空间的一组正交基及A的列向量在这组基下的坐标(R矩阵)

但在求特征值时，注意只有方阵才有特征向量, 不要弄错概念，闹出长方阵也有特征向量

'''

import numpy as np
import sys
import pdb

def QR(A, united=False):
    A_shape = A.shape

    Q = np.zeros(A_shape)    # m*n matrix
    R = np.zeros((A_shape[1], A_shape[1]))  # n*n matrix

    R[0,0] = np.sqrt(np.dot(A[:,0], A[:,0]))
    Q[:,0] = A[:,0] / R[0,0]  #标准化一下

    for col_num in range(1, A.shape[1]):
        for i in range(col_num):
            R[i, col_num] = np.dot(A[:,col_num], Q[:,i])/np.dot(Q[:,i], Q[:,i])

        Q[:,col_num] = A[:,col_num] - np.dot(Q[:,:col_num], R[:col_num, col_num])
        ql = 1
        if united:
            ql = np.sqrt(np.dot(Q[:,col_num], Q[:,col_num]))
        R[col_num,col_num] = ql
        if ql != 0:
            Q[:,col_num] /= ql

    return (Q,R)


A=np.array([[1,2,3], [3,4,5], [1,2,4]])

A = np.array([[1,2,3,2],
              [1,2,7,5],
              [4,9,2,6],
              [6,1,5,8]]);

A = np.array([[1,2,3,2,1],
              [1,2,7,5,2],
              [4,9,2,6,3],
              [6,1,5,8,4]]);

#下面这个矩阵，误差就比较大了
A = np.array([[1,2,3,2,1,1],
              [1,2,7,5,2,2],
              [4,9,2,6,3,3],
              [6,1,5,8,4,4]]);

A = np.array([[1,2,3],
              [1,2,7],
              [4,9,2],
              [6,1,5]]);

A = np.array([[3,-2,4],
              [-2,6,2],
              [4,2,3]]);

if __name__=='__main__':
    Q,R = QR(A)

    print "A:"
    print A
    print "\nQ:"
    print Q
    print "\nR:"
    print R

    print "\nQ*R:"
    print np.dot(Q, R)
