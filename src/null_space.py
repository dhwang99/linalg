#encoding: utf8
'''
'''

import numpy as np
import sys
import pdb

'''
QR可以求矩阵A列空间的标准正交基. 不过要求各列不相关

nullspace计算A零空间的一组标准正交基

cal_nullspace 算法要求A是方阵。准备改为非方阵
'''

from qr import QR

def cal_nullspace(A):
    A = A.copy()
    epsilon = 1e-9
    A_shape = A.shape
    
    # cal main elements first
    for i in range(A_shape[0] - 1):
        if np.abs(A[i, i]) <= epsilon:
            for j in range(i+1, A_shape[0]):
                if np.abs(A[j,i]) > epsilon:  #交换两行。不影响结果
                    tmp = A[j,:]
                    A[j,:] = A[i]
                    A[i,:] = tmp

        if np.abs(A[i,i]) <= epsilon:
            continue

        for j in range(i+1, A_shape[0]):
            t = 1.0 * A[j,i]/A[i,i]
            A[j,:] = A[j,:] -  t * A[i,:];

    #standard
    for i in range(A_shape[0]):
        if np.abs(A[i,i]) > epsilon:
            A[i,:] = A[i,:] * (1.0/A[i,i]) #the main element's value equal to 1.
    
    #计算自由列个数. 即主元为0的列的个数 
    #要加类型，要不然取下标时会报错
    free_eles = np.zeros(A_shape[0], dtype='int') 
    cn = 0
    eigv = np.zeros(A_shape[0])
    for i in range(A_shape[0]):
        if np.abs(A[i,i]) < epsilon:
            free_eles[cn] = i
            cn += 1

    if cn == 0:
        return None

    ns = np.zeros((A_shape[0], cn))

    for i in range(cn):
        ns[free_eles[i], i] = 1
        for j in range(A_shape[0] - 1, -1, -1):
            if np.abs(A[j,j]) > epsilon:
                ns[j, i] = 0 - np.dot(A[j, j+1:], ns[j+1:, i])

    if cn >= 2:    #正交化
        q,r = QR(ns, united=True)
        ns = q
    else:
        sq = np.sqrt(np.dot(ns[:,0], ns[:,0]))
        ns /= sq
        
    return ns 
