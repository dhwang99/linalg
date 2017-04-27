#encoding: utf8

import numpy as np
import sys
import pdb

'''
QR2对QR里的正交化特征向量进行了标准化
用QR2进行QR分解结果可以对对称矩阵求得特征值 
用QR里的QR算法，不能收敛. 可以试跑一下

对QR分解还需要实现一下更高效的算法

A = QR

Q = A * R.inv
A1 = R * A * R.inv = R*Q
则A1相似于A, 则A与A1有相同的特征值 

参考笔记里的内容

求特征向量，用的是解 (A-lambda*I)x = 0, 并对重根进行了正交化。这个比较麻烦
'''

from qr import QR 
from lu import LU
from null_space import cal_nullspace

'''
QR分解求特征值. 原理如上
'''
def cal_eigvals_by_qr(A):
    #构造对称矩阵
    A1 = A.copy()
    epsilon = 1e-9
    #1. 求特征值
    t = 0
    while True:
        Q1,R1 = QR(A1, united=True)
        A1 = np.dot(R1, Q1)
        t = 0
    
        for i in range(1, A.shape[1]):
            for j in range(0, i):
                t += np.abs(A1[i,j])
        
        #print t
        if t <= epsilon:
            break
    
    #print A1
    return np.diag(A1)



def eig(A):
    #1. 求特征值
    es = cal_eigvals_by_qr(A)
    es = np.sort(es)
    eig_vals = es[::-1]  #逆序一下
    A_shape = A.shape
    #2. 求特征向量
    eig_vectors=np.zeros((A.shape[0], A.shape[0]))
    preval = None
    curid=0
    for eigval in eig_vals:
        if preval == eigval:
            continue
        A_eig = A - np.eye(A_shape[0]) * eigval
        cur_vectors = cal_nullspace(A_eig)
        preval = eigval
    
        for i in range(cur_vectors.shape[1]):
            eig_vectors[:, curid] = cur_vectors[:, i]  
            curid += 1

    return (eig_vals,eig_vectors)


'''
例子
'''
A = np.array([[3,-2,4],
              [-2,6,2],
              [4,2,3]]);

A = np.array([[1,2,3,2],
              [1,2,7,5],
              [4,9,2,6],
              [6,1,5,8]]);

A=np.dot(A.T, A)

if __name__=='__main__':
    e1,v1 = eig(A)
    
    print "eigvals:"
    print e1
    
    print "eigvectors:"
    print v1 
    
    print "v0.v1, v0.v2, v1.v2"
    print np.dot(v1[:,0], v1[:,1]), \
          np.dot(v1[:,0], v1[:,2]), \
          np.dot(v1[:,1], v1[:,2])
    
    print "A*v0/e0"
    print np.dot(A, v1[:,0])/e1[0]
    
    print "A*v1/e1"
    print np.dot(A, v1[:,1])/e1[1]
    
    print "A*v2/e2"
    print np.dot(A, v1[:,2])/e1[2]
    
    e,v = np.linalg.eig(A)
    
    print "e:"
    print e
    
    print "v:"
    print v
