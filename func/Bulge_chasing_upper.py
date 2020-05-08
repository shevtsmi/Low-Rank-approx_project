#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 14:27:06 2020

@author: zyl
"""


import numpy as np
import math

def sign(a):
    if(a == 0):
        return 0
    if(a > 0):
        return 1
    return -1


def formrot(a, b):
    c = 0
    s = 0
    if(a == 0 and b == 0):
        print("Error with a,b in formrot")
        return 0, 0
    if(b == 0):
        c = a
        s = 0
    else:
        if(math.fabs(a) <= math.fabs(b)):
            r = math.fabs(a / b)
            factor = math.sqrt(1 + r**2)
            c = sign(a) * r / factor
            s = - sign(b) / factor
        else:
            r = math.fabs(b / a)
            factor = math.sqrt(1 + r**2)
            c = sign(a) / factor
            s = - sign(b) * r / factor 
    return c, s


# i < j <= size - 1   -- индексы для ротации 
# c, s -- параметры, задающие ротацию
def Givens_left(c, s, size, i, j):
    G = np.eye(size, size)
    G[i][i] = c
    G[j][j] = c
    G[i][j] = -s
    G[j][i] = s
    return G

def Givens_right(c, s, size, i, j):
    G = np.eye(size, size)
    G[i][i] = c
    G[j][j] = c
    G[i][j] = s
    G[j][i] = -s
    return G

# Input:
#   b -- вектор диагональной матрицы (Sigma = Diag(b))
#   z -- вектор, относительно которого происходит нормировка  (в статье обозначен как x)
#   n -- размер векторов


# Output:
#  Q  W  Такие, что : Q * Sigma * W -- верхняя треугольная, W^T * z = || z || * e_1

def Bulge_chasing_upper(b_, z_, n):
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
    c, s = formrot(z[n - 2], z[n - 1])
    z[n - 2] = c * z[n - 2] - s * z[n - 1]
    z[n - 1] = 0 #s * z[n - 2] + c* z[n - 1]   # === 0
    g[n - 2] = s * b[n - 2]
    b[n - 2] = c * b[n - 2]
    xi[n - 2] = -s * b[n - 1]
    b[n - 1] = c * b[n - 1]
    
    W = Givens_right(c, s, n, n - 2, n - 1)
    #---------------------------------
    c, s = formrot(b[n - 2], xi[n - 2])
    temp = b[n - 2]
    b[n - 2] = c * temp - s * xi[n - 2]
    xi[n - 2] = 0 #s * temp + c * xi[n - 2]   # === 0
    
    temp = g[n - 2]
    g[n - 2] = c * temp - s * b[n - 1]
    b[n - 1] = s * temp + c * b[n - 1]
    
    Q = Givens_left(c, s, n, n - 2, n - 1)

    #iterations
    
    for i in range(n - 2, 0, -1):   ## n-2, 0
        if(z[i] == 0):
            continue
        c, s = formrot(z[i - 1], z[i])
        temp = z[i]
        z[i] = 0    # s * z[i - 1] + c * temp    # === 0
        z[i - 1] = c * z[i - 1] - s * temp
        xi[i - 1] = -s * b[i]
        b[i] = c * b[i]
        g[i - 1] = s * b[i - 1]
        b[i - 1] = c * b[i - 1]
        W = W.dot(Givens_right(c, s, n, i - 1, i))

        for j in range(i, n - 1):
            # first reduction step
            c, s = formrot(b[j - 1], xi[j - 1])
            temp = b[j - 1]
            b[j - 1] = c * temp - s * xi[j - 1]
            xi[j - 1] = 0    # s * temp + c * xi[j - 1] # === 0
            
            temp = g[j - 1]
            g[j - 1] = c * temp - s * b[j]
            b[j] = s * temp + c * b[j]
            
            temp = g[j]
            g[j] = c * temp
            l[j - 1] = -s * temp
            
            Q = Givens_left(c, s, n, j - 1, j).dot(Q)

            #second reduction rotation
            c, s = formrot(g[j - 1], l[j - 1])

            temp = b[j]
            b[j] = c * temp - s * g[j]
            g[j] = s * temp + c * g[j]
            
            temp = g[j - 1]
            g[j - 1] = c * temp - s * l[j - 1]
            l[j - 1] = 0     #s * temp + c * l[j - 1]   # === 0
            
            xi[j] = -s * b[j + 1]
            b[j + 1] = c * b[j + 1]
            W = W.dot(Givens_right(c, s, n, j, j + 1))

        #Final rotation
        c, s = formrot(b[n - 2], xi[n - 2])
        temp = b[n - 2]
        b[n - 2] = c * temp - s * xi[n - 2]
        xi[n - 2] = 0   # s * b[n - 2] + c * xi[n - 2]  # === 0
        
        temp = g[n - 2]
        g[n - 2] = c * temp - s * b[n - 1]
        b[n - 1] = s * temp + c * b[n - 1]
        
        Q = Givens_left(c, s, n, n - 2, n - 1).dot(Q)
        
    return Q, W