#!/usr/bin/env python3
# -*- coding: utf-8 -*-



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
    c, s = formrot(z[n - 2], z[n - 1])
    z[n - 2] = c * z[n - 2] - s * z[n - 1]
    z[n - 1] = 0 #s * z[n - 2] + c* z[n - 1]   # === 0
    g[n - 2] = s * b[n - 2]
    b[n - 2] = c * b[n - 2]
    xi[n - 2] = -s * b[n - 1]
    b[n - 1] = c * b[n - 1]
    
    W = Givens_right(c, s, n, n - 2, n - 1) 
    """
    print("start 1\n")
    print(Q.dot(Sigma.dot(W)))  
    print("\n")
    print(b, "\n")
    print(xi, "\n")
    print(g, "\n")
    """
    #---------------------------------
    c, s = formrot(b[n - 1], g[n - 2])
    s = -s
    temp = b[n - 2]
    b[n - 2] = c * temp - s * xi[n - 2]
    xi[n - 2] = s * temp + c * xi[n - 2]   
    
    temp = g[n - 2]
    g[n - 2] = 0 #c * temp - s * b[n - 1]   # === 0
    b[n - 1] = s * temp + c * b[n - 1]
    
    Q = Givens_left(c, s, n, n - 2, n - 1)
    """
    print("start 2\n")
    print(Q.dot(Sigma.dot(W)))  
    print("\n")
    print(b, "\n")
    print(xi, "\n")
    print(g, "\n")
    """
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
        
        
        c, s = formrot(b[i], g[i - 1])
        s = -s

        temp = b[i - 1]
        b[i - 1] = c * temp - s * xi[i - 1]
        xi[i - 1] = s * temp + c * xi[i - 1] # === 0

        temp = g[i - 1]
        g[i - 1] = 0  #c * temp - s * b[i]    # === 0
        b[i] = s * temp + c * b[i]

        """
        temp = g[j]
        g[j] = c * temp
        l[j - 1] = -s * temp
        """
        Q = Givens_left(c, s, n, i - 1, i).dot(Q)
        """
        print("beg\n")
        print(Q.dot(Sigma.dot(W)))  
        print("\n")
        print(b, "\n")
        print(xi, "\n")
        print(g, "\n")
        """
        
        for j in range(i, n - 2):
            # first lower reduction step
            # print("j ============ {}".format(j))
            
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
            """
            print("after all step i = {}, j = {} \n".format(i, j))
            print(Q.dot(Sigma.dot(W)))  
            print("\n")
            print(b, "\n")
            print(xi, "\n")
            print(g, "\n")
            print(l, "\n")
            """
        
        
        
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
        """
        print("predfinal\n")
        print(Q.dot(Sigma.dot(W)))  
        print("\n")
        print(b, "\n")
        print(xi, "\n")
        print(g, "\n")
        print(l, "\n")
        """
        
        c, s = formrot(b[n - 2], g[n - 2])
        temp = b[n - 2]
        b[n - 2] = c * temp - s * g[n - 2]
        g[n - 2] = 0    #s * temp + c * xi[n - 2]   # === 0
        
        temp = xi[n - 2]
        xi[n - 2] = c * temp - s * b[ n - 1]
        b[n - 1] = s * temp + c * b[n - 1]

        W = W.dot(Givens_right(c, s, n, n - 2, n - 1))
        
        """
        print("final\n")
        print(Q.dot(Sigma.dot(W)))  
        print("\n")
        print(b, "\n")
        print(xi, "\n")
        print(g, "\n")
        print(l, "\n")
        
        #Final rotation
        c, s = formrot(b[n - 2], xi[n - 2])
        temp = b[n - 2]
        b[n - 2] = c * temp - s * xi[n - 2]
        xi[n - 2] = 0   # s * b[n - 2] + c * xi[n - 2]  # === 0
        
        temp = g[n - 2]
        g[n - 2] = c * temp - s * b[n - 1]
        b[n - 1] = s * temp + c * b[n - 1]
        
        Q = Givens_left(c, s, n, n - 2, n - 1).dot(Q)
        """

    return Q, W












































