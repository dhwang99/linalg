#encoding: utf8

import numpy as np
import sys
import pdb

'''
求伪逆用三种方法

1. QR分解
    A=QR, 则 inv(A)=inv(R)*Q.T
     AXA = QR * inv(R) *Q.T * QR=Q*Q.T * QR=QR=A
     XAX=inv(R)*Q.T * QR * inv(R)*Q.T = inv(R)*Q.T

     XA = inv(R) * Q.T * Q * R = I
     AX = Q* R* inv(R) * Q.T = Q*Q.T, 为一对称矩阵 

2. svd分解
       +   +   +
    A=U * S * V,
             +          +     +
    inv(A)=(V).T * inv(S) * (U).T

3. 直接求解
    
'''

from qr import QR
from inv import inv
from svd import svd

def pinv_by_qr(A):
    q,r = QR(A, united=True)
    r_inv = inv(r)
    A_pinv = np.dot(r_inv, q.T)

    return A_pinv

'''
用svd来解pinv, 因涉及对S求逆，这个要求返回简化 SVD分解结果，S为r*r的对角阵
'''
def pinv_by_svd(A):
    U,S,V = svd(A, retSimple=True)
    S = inv(S)
    A_pinv = np.dot(np.dot(V.T, S), U.T)

    return A_pinv

A=np.array([[1,2,3], 
           [4,5,6], 
           [1,1,5]], dtype='float');

A=np.array([[1,2,3], 
           [1,0,1],
           [4,5,6], 
           [1,1,5]], dtype='float');

pinv_fun=pinv_by_qr
pinv_fun=pinv_by_svd

if __name__=='__main__':
    A_pinv = pinv_fun(A)

    print "A:"
    print A

    print "pinv(A):"
    print A_pinv

    print "A*A_pinv:"
    print np.dot(A, A_pinv)

    print "A_pinv*A:"
    print np.dot(A_pinv, A)

    print "A*A_pinv*A:"
    print np.dot(np.dot(A, A_pinv), A)

    print "A_pinv*A*A_pinv:"
    print np.dot(np.dot(A_pinv, A), A_pinv)
