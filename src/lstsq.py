#encoding: utf8

import numpy as np
import sys
import pdb

'''
Ax = x_head
最小二乘法的实现.
三种实现方法：
1. 正规方程解法。
  取y为b的A平面的投影，则 A 与(b - Ay)正交, 此时， |b - Ay|最小（其实不一定。还有一个最小值。 比如:
  A=[1 0 0 0].T, b=[1  1 1 1], Ax = b
  例子有点不对。还有一个 伪逆的求解方法
    
    A.T * (b - Ay) = 0
    A.T * b - A.T * A * y = 0
    y = inv(A.T * A) * A.T * b


2. QR分解。
正规方程在某些时候方程是病态的。A.T * A，会放大误差:即很少的差异，会带来比较大的误差。条件数大
于是，如果 A的列不相关。则可以用QR分解来解
A=QR
y=inv(R) * Q.T * b
Ay = QR*inv(R) * Q.T * b = Q*Q.T * b
Q*Q.T*b为b在Q上的正交投影 (?看讲义), 即为所求的解
因为A的列线性无关，Ax=b有唯一的最小二乘解。于是上式成立 (这个需要证明。参考线性代数及其应用书）
实际解时，一般是解

Ry = Q.T*b, 因为R本身是一个上三解矩阵

3. SVD解法
A+= U<sup>+</sup>
A_inv = 

'''

from inv import inv

#y = a + bx + cx**2

x = np.array([0,1,2,3,4,5])
y = np.array([2.1, 2.9, 4.15, 4.98, 5.5, 6])

x2 = np.power(x,2)
x0 = np.ones(x.shape[0])

#A = np.vstack((x0, x, x2)).T
#为了和后面的polyfit一致. polyfit高次在前
A = np.vstack((x2, x, x0)).T

#正规方程求 lstsq
#inv(A'A) A.T*b
x_head = np.dot(np.dot(inv(np.dot(A.T, A)), A.T), y)

errors = np.sum(np.abs(np.dot(A, x_head) - y)) / np.sum(np.abs(y))

print "errors by inv(A.T*A)*A.T*b:", errors

#QR分解求 lstsq 
from qr import QR

Q,R = QR(A, united=True)
x_head_byQR = np.dot(np.dot(inv(R),Q.T),  y)

errors = np.sum(np.abs(np.dot(A, x_head_byQR) - y)) / np.sum(np.abs(y))
print "errors by inv(R)*Q.T*b:", errors

#SVD分解求 lstsq 
from pinv import pinv_by_svd 
x_pinv = pinv_by_svd(A)
x_head_bySVD = np.dot(x_pinv,  y)

errors = np.sum(np.abs(np.dot(A, x_head_bySVD) - y)) / np.sum(np.abs(y))
print "errors by x_head_bySVD*b:", errors

#多项式拟合
z = np.polyfit(x, y, 2)
p = np.poly1d(z)
errors = np.sum(np.abs(np.dot(A, p) - y)) / np.sum(np.abs(y))
print "errors by poly1d:", errors

#plotting
# export MPLBACKEND="agg"  in shell. or use follow
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

import matplotlib.pyplot as plt
#pdb.set_trace()
xp = np.linspace(-1, 6, 100)
plt.plot(x, y, '.', xp, p(xp))
plt.savefig('images/lstsq.png', format='png')
