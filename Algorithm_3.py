#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas
import math
import os
#cimport scipy.linalg.cython_lapack
os.getcwd()


from func.Bulge_chasing_lower import Bulge_chasing_lower
from func.Bulge_chasing_upper import Bulge_chasing_upper
from func.Select_Index import Select_Columns, Select_Rows, Select_Ind
from func.Bidiag_Rotations import house_bidiag
from func.Tridiag_Rotations import Tridiagonal_chase
from func.Summation_Alg import GetCoFromEigen

def thin_SVD(B):
    U_, S, V_ = np.linalg.svd(B, full_matrices=False)
    Sigma_ = np.diag(S)
    return U_, Sigma_, V_

def Algorithm_3(A, k):
   
    m = A.shape[0]
    n = A.shape[1]
    
    B = A.copy()
    I = []
    J = []

    i_t = -1
    j_t = -1
    for t in range(k):
        minRatio = math.inf
        U, sig, V = np.linalg.svd(B, full_matrices=False)
        Sigma = np.diag(sig)
        
        for i in range(m):    # 0, ... , (m - 1)
            for j in range(n):  # 0, ... , (n - 1)
            
                print("{}/{} ; {}/{} ; {}/{}".format(t, k, i, m, j, n))
                x = Sigma.dot(V[:, j])
                    # check if B(i, j) == 0
                
                if(B[i][j] == 0):
                    continue
                
                y = 1/B[i][j] * Sigma.dot(U[i, :].T)
                

                Giv_1 = Bulge_chasing_lower(sig, x, len(sig))
                
                D_1 = Giv_1.apply_T(Sigma - np.outer(x, y))
                
                D_2 = Tridiagonal_chase(D_1, full=False)

                D_3 = house_bidiag(D_2, full=False)

                __, sing, __  = np.linalg.svd(D_3, full_matrices=False)
                
                # Summation Algorithm
                eigen = np.diag(sing) @ sing
                # print("==================")
                # print(eigen)
                coeff = GetCoFromEigen(eigen)
                up   = coeff[min(m,n) - k + t - 1]
                down = coeff[min(m,n) - k + t]
                if down == 0:
                    r = math.inf
                else:
                    r = up/down
                
                
                if(r < minRatio and B[i][j] != 0 and i not in I and j not in J):
                    
                    # print("================")
                    # print("t = ", t)
                    # print("i_t = {}, j_t = {}, \n I = \n{}\n J = \n{}\n".format(i, j, I , J))
                    i_t = i
                    j_t = j
                    minRatio = r
                
        # print("i_t = {} ; j_t = {}\n".format(i_t, j_t))
        I.append(i_t)
        J.append(j_t)
        I.sort()
        J.sort()
        
        # print("B[i_t][j_t] = ", B[i_t][j_t])
        if B[i_t][j_t] !=0:
            B = B - 1/B[i_t][j_t] * np.outer(B[:, j_t], B[i_t, :])
    
    C  = Select_Columns(A, J)
    R  = Select_Rows(A, I)
    U = np.linalg.pinv(C) @ A @ np.linalg.pinv(R)
    
    return C, U, R, I, J