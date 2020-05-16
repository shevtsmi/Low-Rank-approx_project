#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
os.getcwd()
np.set_printoptions(suppress=True)


def make_house_vec(x):
    n = x.shape[0]
    dot_1on = x[1:].dot(x[1:])

    # v is our return vector; hack on v[0]
    v = np.copy(x)
    v[0] = 1.0
    
    if dot_1on < np.finfo(float).eps:
        beta = 0.0
    else:
        # apply Parlett's formula
        norm_x= np.sqrt(x[0]**2 + dot_1on)
        if x[0] <= 0:
            v[0] = x[0] - norm_x
        else:
            v[0] = -dot_1on / (x[0] + norm_x)
        beta = 2 * v[0]**2 / (dot_1on + v[0]**2)
        v = v / v[0]
    return v, beta

def full_house(n, col, v, beta):
    full = np.eye(n)
    full[col:, col:] -= beta * np.outer(v,v)
    return full





# Input : произвольная квадратная матрица A (n x n) 

# Output: U, B, Vt
     # U, Vt -- ортогональные квадратные матрицы размера (n x n)
     # B     -- верхняя треугольная матрица (n x n) вида:
         
     #     * * 0 0 0 0
     #     0 * * 0 0 0 
     #     0 0 * * 0 0 
     #     0 0 0 * * 0
     #     0 0 0 0 * *
     #     0 0 0 0 0 *
 
    # Так, что выполнено равенство: U * A * V_t = B

def house_bidiag(A):
    m = A.shape[0]
    n = A.shape[1]
    assert m >= n
    U,Vt = np.eye(m), np.eye(n)
    
    for col in range(n):
        v, beta = make_house_vec(A[col:,col])
        A[col:,col:] = (np.eye(m-col) - beta * np.outer(v,v)).dot(A[col:,col:])
        Q = full_house(m, col, v, beta)
        U = U.dot(Q)
        
        if col <= n-2:
            v,beta = make_house_vec(A[col,col+1:].T)
            A[col:,col+1:] = A[col:, col+1:].dot(np.eye(n-(col+1)) - beta * np.outer(v,v))
            P = full_house(n, col+1, v, beta)
            Vt = P.dot(Vt)
    return U, A, Vt








