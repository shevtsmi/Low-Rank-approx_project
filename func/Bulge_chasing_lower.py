#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import math
import os
os.getcwd()

from func.Givens_Rotation import formrot, Givens_left, Givens_right

# Input:
#   b -- вектор диагональной матрицы (Sigma = Diag(b))
#   z -- вектор, относительно которого происходит нормировка  (в статье обозначен как x)
#   n -- размер векторов
# -
#      * * * * * *                       * 0 0 0 0 0           --- z
     
#      * 0 0 0 0 0                       * 0 0 0 0 0
#      0 * 0 0 0 0      Q*diag(b)*W      * * 0 0 0 0
#      0 0 * 0 0 0        ====>          0 * * 0 0 0           --- Нижняя бидиагональная
#      0 0 0 * 0 0                       0 0 * * 0 0
#      0 0 0 0 * 0                       0 0 0 * * 0
#      0 0 0 0 0 *                       0 0 0 0 * *

# Output:
#  Q  W  Такие, что : Q * Sigma * W -- нижняя треугольная бидиагональная, W^T * z = || z || * e_1

def Bulge_chasing_lower(b_, z_, n):
    b = b_.copy()
    z = z_.copy()
    temp = 0
    g  = np.zeros(n - 1)
    xi = np.zeros(n - 1)
    l  = np.zeros(n - 2)
    Q = np.eye(n)
    W = np.eye(n)
    # start
    #---------------------------------
    
    if(z[n - 1] != 0):
        c, s = formrot(z[n - 2], z[n - 1])
        z[n - 2] = c * z[n - 2] - s * z[n - 1]
        z[n - 1] = 0 #s * z[n - 2] + c* z[n - 1]   # === 0
        g[n - 2] = s * b[n - 2]
        b[n - 2] = c * b[n - 2]
        xi[n - 2] = -s * b[n - 1]
        b[n - 1] = c * b[n - 1]
        
        W = Givens_right(c, s, n, n - 2, n - 1) 

        c, s = formrot(b[n - 1], g[n - 2])
        s = -s
        temp = b[n - 2]
        b[n - 2] = c * temp - s * xi[n - 2]
        xi[n - 2] = s * temp + c * xi[n - 2]   
        
        temp = g[n - 2]
        g[n - 2] = 0 #c * temp - s * b[n - 1]   # === 0
        b[n - 1] = s * temp + c * b[n - 1]
        
        Q = Givens_left(c, s, n, n - 2, n - 1)

    #iterations
    
    for i in range(n - 2, 0, -1):   ## n-2, 0, -1
        if(z[i] == 0):
            continue
        c, s = formrot(z[i - 1], z[i])
        temp = z[i]
        z[i] = 0    # s * z[i - 1] + c * temp    # === 0
        z[i - 1] = c * z[i - 1] - s * temp
        
        g[i - 1] = s * b[i - 1]
        b[i - 1] = c * b[i - 1]
        
        xi[i - 1] = -s * b[i]
        b[i] = c * b[i]
        
        l[i - 1] = -s * xi[i]
        xi[i] = c * xi[i]
        
        W = W.dot(Givens_right(c, s, n, i - 1, i))
        
        
        if(g[i - 1] != 0):
            c, s = formrot(b[i], g[i - 1])
            s = -s
    
            temp = b[i - 1]
            b[i - 1] = c * temp - s * xi[i - 1]
            xi[i - 1] = s * temp + c * xi[i - 1] # === 0
    
            temp = g[i - 1]
            g[i - 1] = 0  #c * temp - s * b[i]    # === 0
            b[i] = s * temp + c * b[i]
    
            Q = Givens_left(c, s, n, i - 1, i).dot(Q)
 
        for j in range(i, n - 2):
            
            if(l[j - 1] != 0):
                c, s = formrot(xi[j  -1], l[j - 1])
    
                temp = b[j]
                b[j] = c * temp - s * xi[j]
                xi[j] = s * temp + c * xi[j]
                
                g[j] = -s * b[j + 1]
                b[j + 1] = c * b[j + 1]
                
                temp = l[j - 1]
                l[j - 1] = 0  #s * xi[j - 1] + c * temp  # === 0
                xi[j - 1] = c * xi[j - 1] - s * temp
    
                Q = Givens_left(c, s, n, j, j + 1).dot(Q)
            
                # second upper reduction step
                c, s = formrot(b[j], g[j])
                
                temp = b[j]
                b[j] = c * temp - s * g[j]
                g[j] = 0   #s * temp + c * g[j]        #=== 0
                
                temp = xi[j]
                xi[j] = c * temp - s * b[j + 1]
                b[j + 1] = s * temp + c * b[j + 1]
                
                l[j] = -s * xi[j + 1]
                xi[j + 1] = c * xi[j + 1]
                
                W = W.dot(Givens_right(c, s, n, j, j + 1))
        
        if(l[n - 2 - 1] != 0):
            c, s = formrot(xi[n - 2 -1], l[n - 2 - 1])
    
            temp = b[n - 2]
            b[n - 2] = c * temp - s * xi[n - 2]
            xi[n - 2] = s * temp + c * xi[n - 2]
    
            g[n - 2] = -s * b[n - 2 + 1]
            b[n - 2 + 1] = c * b[n - 2 + 1]
    
            temp = l[n - 2 - 1]
            l[n - 2 - 1] = 0  #s * xi[j - 1] + c * temp  # === 0
            xi[n - 2 - 1] = c * xi[n - 2 - 1] - s * temp
    
            Q = Givens_left(c, s, n, n - 2, n - 2 + 1).dot(Q)
            
            c, s = formrot(b[n - 2], g[n - 2])
            temp = b[n - 2]
            b[n - 2] = c * temp - s * g[n - 2]
            g[n - 2] = 0    #s * temp + c * xi[n - 2]   # === 0
            
            temp = xi[n - 2]
            xi[n - 2] = c * temp - s * b[ n - 1]
            b[n - 1] = s * temp + c * b[n - 1]
    
            W = W.dot(Givens_right(c, s, n, n - 2, n - 1))

    return Q, W












































