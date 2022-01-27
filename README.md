### 线性代数中一些基础算法的python实现

> 建议采用 Gilbert Strang 的《Introduction to Linear Algebra》 教材，有对应的中文版
> 
> 用python实现了常用的线性代数的基础算法.python2.7跑例子可以通过。很多细节没有考虑

> 主要用来学习线性代数的算法原理. eig求解时，用到迭代算法.

> 在代码里也有一些注释

#### 1. 方阵求逆
>    采用类似LU分解的方法求矩阵的逆。

>    [inv.py](src/inv.py)

#### 2. LU分解
>    高斯消元法求解LU. 要求A可逆

>    [lu.py](src/lu.py)

#### 3. 求方阵A的左零空间的基
>   主要用来求解 Ax = 0的基础解。类似高斯消元法，求解给出了零空间的一组标准正交基. A的行空间和零空间正交

>   [null_space.py](src/null_space.py)

#### 4. 解线性方程组
>   求解 Ax = b. 使用LU分解求解。 要求A是非奇异的。

>   [solve.py](src/solve.py)

#### 5. QR分解
>   使用  Gram-Schmidt正交化对矩阵进行QR分解 

>   [qr.py](src/qr.py)

#### 6.  QR分解求解矩阵的特征值和特征向量
>    考虑了重根的情况。但没有考虑虚根等等

>    [eig_by_qr.py](src/eig_by_qr.py)

#### 7. 幂法、逆幂法求解最大、最小特征值 
>    [eig_by_pow_inverse.py](src/eig_by_pow_inverse.py)

#### 8. 对角化
>   A=PDP<sup>-1</sup>分解算法 

>   对矩阵A先求特征值和向量，然后进行对角化 

>    [diagonalization.py](src/diagonalization.py)

#### 9. SVD分解
>    先用eig_by_qr方法求解A.T * A的特征值和特征向量，然后进行SVD分解 

>    SVD分解应用比较广，实际并不用eig_by_qr来求解(方程容易病态), 用迭代法比较多

>    [svd.py](src/svd.py)

#### 10. 求伪逆 
>    实现了两个算法。 QR分解和SVD求伪逆. 其中QR分解要求A阵列线性无关

>    [pinv.py](src/pinv.py)

#### 11. 最小二乘法
>    使用正规方程、QR分解、SVD分解、多项式拟合求解最小二乘 

>    SVD分解直接用的SVD求伪逆函数

>    [lstsq.py](src/lstsq.py)
