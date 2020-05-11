#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import math
import os
os.getcwd()


from func.Bulge_chasing_lower import Bulge_chasing_lower
from func.Bulge_chasing_upper import Bulge_chasing_upper
from func.Select_Index import Select_Columns, Select_Rows, Select_Ind

def GetCoFromEigen(eigen):
    n = len(eigen)
    ans = []
    S = np.zeros((n, n+1), dtype = np.complex)
    S[0,1] = eigen[0]
    for i in range(n-1):
        S[i, 0] = 1
    for i in range(1, n):
        for j in range(1, i+2):
            S[i, j] = S[i-1, j] + eigen[i]*S[i-1,j-1]
    ans.append(complex(1.0,0))
    for i in range(1,n+1):
        ans.append(S[n-1, i])
    return ans


# A in R^{m x n};    select any 1 <= k < m
def GetColumnSubset(A, k):
    S = []
    B = A.copy()
    n = A.shape[1]
    m = A.shape[0]
    #print("M, n = ", m, n)
    for t in range(k):
        U, sig, til = np.linalg.svd(B, full_matrices = False)
        minRatio = math.inf
        for i in range(n):
            b = []
            for l in range(m):
                b.append(B[l,i])
            q = U.T @ b / np.linalg.norm(b)
            Q, W = Bulge_chasing_lower(sig, q, len(sig))
            temp = Q
            Q = W.T
            W = temp.T
            
            
            D = Q @ np.diag(sig) @ W
            e1 = Q @ q
            
            til, sing, til1  = np.linalg.svd((np.eye(len(e1)) - e1 @ e1.T)@D)
            
            
            eigen = np.diag(sing) @ sing
            coeff = GetCoFromEigen(eigen)
            up = coeff[m-k+t- 1]
            down = coeff[m-k+t]
            ratio = up/down
            if ratio < minRatio:
                minRatio = ratio
                i_t = i
        S.append(i_t)
        S.sort()
        # print("t = {}\n".format(t))
        # print("S = {}\n".format(S))
        # print("A = \n", A)
        
        
        Ans = Select_Columns(A, S)
        # print("Ans = \n", Ans)
        # print("====================")
        Q, til  = np.linalg.qr(Ans)
        B = A - Q @ Q.T @ A
    return S

def Algorithm_2(A, k):
    Cs = GetColumnSubset(A, k)
    C  = Select_Columns(A, Cs)
    
    Rs = GetColumnSubset(A.T, k)
    R  = Select_Rows(A, Rs)
    U = np.linalg.pinv(C) @ A @ np.linalg.pinv(R)
    #U = np.linalg.inv(Select_Ind(A, Rs, Cs))
    return C, U, R, Cs, Rs



