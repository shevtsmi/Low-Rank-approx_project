#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import os
os.getcwd()

from func.Givens_Rotation import formrot, Givens_left, Givens_right

# Input:
#   D -- квадратная матрица размеров n x n со структурой:
        
        # 1. В первой строке есть ненулевые элементы
        # 2. На главной диагонали есть ненулевые элементы
        # 3. На диагонали, которая над главной есть ненулевые элементы
        
        #       * * * * * *
        #       0 * * 0 0 0 
        #       0 0 * * 0 0
        #       0 0 0 * * 0
        #       0 0 0 0 * *
        #       0 0 0 0 0 *   

# Output: Q, B, W
#  Тридиагональная матица B размеров n x n вида:
    
                            #      * * * 0 0 
                            #      0 * * * 0
                            #      0 0 * * *
                            #      0 0 0 * *
                            #      0 0 0 0 *
    # Две ортогональные матрицы Q, W размеров n x n, которые
    # при умножении дают заданную ротацию:
        
    #     Q * D * W = B

def Tridiagonal_chase(A):
    n = A.shape[0]
    m = A.shape[1]
    if(n != m or  n <= 4):
        print("Error : matrix should be n x n, n > 4")
        
    z = A[0, :].copy()
    b = np.diag(A).copy()
    g = np.diag(A, 1).copy()
    
    Q = np.eye(n)
    W = np.eye(n)
    p = np.zeros(n - 2)
    q = np.zeros(n - 3)
    xi = np.zeros(n - 1)
    
    # прямой ход:
    for i in range(n - 2, 1, -1):
        if(z[i + 1] == 0):
            continue;
        
        c, s = formrot(z[i], z[i + 1])
        W = W.dot(Givens_right(c, s, n, i, i + 1))
        
        temp = z[i + 1]
        z[i + 1] = s * z[i] + c * temp  # === 0
        z[i] = c * z[i] - s * temp
        
        p[i - 1] = s * g[i - 1]
        g[i - 1] = c * g[i - 1]
        
        temp = b[i]
        b[i] = c * temp - s * g[i]
        g[i] = s * temp + c * g[i]
        
        xi[i] = -s * b[i + 1]
        b[i + 1]      = c * b[i + 1]

        for j in range(i + 1, n): # j = i + 1,  .... , n - 2  range(i + 1, n - 1)
        
            # horizontal shift
            if(xi[j - 1] != 0):
                c, s = formrot(b[j - 1], xi[j - 1])
                Q = Givens_left(c, s, n, j - 1, j).dot(Q)
                
                temp = b[j - 1]
                b[j - 1]  = c * temp - s * xi[j - 1]
                xi[j - 1] = s * temp + c * xi[j - 1]    # === 0
                
                temp = g[j - 1]
                g[j - 1] = c * temp - s * b[j]
                b[j]     = s * temp + c * b[j]
                
                if(j + 1 < n):
                    temp = p[j + 1 - 2]
                    p[j + 1 - 2] = c * temp - s * g[j + 1 - 1]
                    g[j + 1 - 1] = s * temp + c * g[j + 1 - 1]
                    
                    if(j + 2 < n):
                        q[j + 2 - 3] = -s * p[j + 2 - 2]
                        p[j + 2 - 2] = c * p[j + 2 - 2]
            
            
            j += 1

            if(j >= n - 1):
                break
            
            # Vertical shift
            
            if(q[j + 1 - 3] != 0):
                c, s = formrot(p[j - 2], q[j + 1 - 3])
                W = W.dot(Givens_right(c, s, n, j, j + 1))
                
                temp = p[j - 2]
                p[j - 2] = c * temp - s * q[j + 1 - 3]
                q[j + 1 - 3] = s * temp + c * q[j + 1 - 3] ### === 0
                
                temp = g[j - 1]
                g[j - 1] = c * temp - s * p[j + 1 - 2]
                p[j + 1 - 2] = s * temp + c * p[j + 1 - 2]
                
                temp = b[j]
                b[j] = c * temp - s * g[j + 1 - 1]
                g[j + 1 - 1] = s * temp + c * g[j + 1 - 1]
                
                xi[j] = - s * b[j + 1]
                b[j + 1] = c * b[j + 1]
            
    B = np.diag(b) + np.diag(g, 1) + np.diag(p, 2) + \
        np.diag(q, 3) + np.diag(xi, -1)
    B[0,:] = z
    return Q, B, W
