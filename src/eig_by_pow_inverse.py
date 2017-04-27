#encoding: utf8

import numpy as np
import sys
import pdb

'''
求特征值的另一个方法. 
通过幂法、反幂法求矩阵的最大、最小特征值
'''

from qr import QR 

'''

'''

A = np.array([[1,2,3,2],
              [1,2,7,5],
              [4,9,2,6],
              [6,1,5,8]]);

A=np.dot(A.T, A)

A = np.array([[3,-2,4],
              [-2,6,2],
              [4,2,3]]);


#构造对称矩阵
A1 = A.copy()
x = 2 * np.random.random_sample(A.shape[0]) - 1
#1. 求特征值
x_max = np.max(x)
epsilon = 1e-9
iter_max = 1000

x1 = x

while True:
    x1 = np.dot(A1, x)
    x1_max = x1[np.argmax(np.abs(x1))]
    dc = np.abs(x_max - x1_max)
    print "diff:", dc
    if np.abs(x_max - x1_max) < epsilon:
        x_max = x1_max
        print "eig value:", x_max 
        break
    else:
        x = x1 / x1_max
        x_max = x1_max

max_eig = x_max 
x = np.dot(A1, x1)
if x[0] * x1[0] < 0:
    max_eig = -max_eig
eigvec_with_maxeig = x1

A1=np.linalg.inv(A)

x = 2 * np.random.random_sample(A.shape[0]) - 1
x[0] = 1
x_max = np.max(x)

while True:
    x1 = np.dot(A1, x)
    x1_max = x1[np.argmax(np.abs(x1))]
    dc = np.abs(x_max - x1_max)
    print "diff:", dc
    if np.abs(x_max - x1_max) < epsilon:
        x_max = x1_max
        print "eig value:", x_max 
        break
    else:
        x = x1 / x1_max
        x_max = x1_max

pdb.set_trace()
min_eig = 1.0/x_max 
x = np.dot(A1, x1)
if x[0] * x1[0] < 0:
    min_eig = -min_eig
eigvec_with_mineig = x1

print "max and min eig:", max_eig, min_eig
print "eigvec with maxeig:", eigvec_with_maxeig
print "eigvec with mineig:", eigvec_with_mineig

e,v = np.linalg.eig(A)
print "eigval and vector from np.linalg.eig"
print "eig:"
print e
print "eig vector:"
print v

'''
min_eig = 2, 通过 np.linalg.eig(A) 得到的值为-2, 相当于把特征向量的矩阵反方向了一下
'''

#2. 求特征向量

#3. 同特征值空间正交化

#4. 得到正交矩阵和对角矩阵
